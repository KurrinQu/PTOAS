# TileLang RMSNorm 对接 PTO 后端方案设计

## 1. TileLang RMSNorm Kernel

讨论 PTO 后端对接之前，先把作为输入的 kernel 摆出来。它的结构有几个要点：

- 外层是 64 个 AICORE 上的 persistent kernel；
- 每个 token 的主体计算放进 `T.SimtVF(threads=threads)` 内，`x_frag` 是 SIMT workitem 的本地 fragment；
- 跨 workitem 的规约由 `T.alloc_reducer + T.finalize_reducer` 完成；
- `d=4096`、`threads=128` 下，每个 workitem 负责 32 个 f32 元素，即 8 个 float4。

完整代码：

```python
def rms_norm_fwd(batch, d, dtype="float32"):
    N_CORES = 64
    TILE = tilelang.next_power_of_2(d)

    # Adaptive threads: keep ~32 elements per thread (8 float4)
    threads = 256 if TILE > 4096 else 128

    N = batch * d

    @T.prim_func
    def main(
        X: T.Buffer((N,), dtype),
        Y: T.Buffer((N,), dtype),
        W: T.Buffer((d,), dtype),
        RSTD: T.Buffer((batch,), "float32"),
        eps: T.float32,
    ):
        n_tokens_per_core = batch // N_CORES

        with T.Kernel(N_CORES) as core_id:
            # Double-buffered shared buffers
            w_ub = T.alloc_shared((d,), dtype)
            x_ub = T.alloc_shared((2, TILE), "float32")
            y_ub = T.alloc_shared((2, TILE), "float32")
            z_rstd_ub = T.alloc_shared((2, 8), "float32")

            # Load weights once
            T.copy(W[:d], w_ub[:d])
            T.ascend_set_flag("MTE2_V", 3)
            T.ascend_wait_flag("MTE2_V", 3)

            # Init double-buffer flags
            T.ascend_set_flag("V_MTE2", 0)
            T.ascend_set_flag("MTE3_V", 0)
            T.ascend_set_flag("V_MTE2", 1)
            T.ascend_set_flag("MTE3_V", 1)

            for t in T.serial(n_tokens_per_core):
                base = (t * N_CORES + core_id) * d
                eid = t % 2

                # MTE2: load x[token] from GM to UB
                T.ascend_wait_flag("V_MTE2", eid)
                T.copy(X[base : base + d], x_ub[eid, :d])
                T.ascend_set_flag("MTE2_V", eid)

                # VEC: compute RMSNorm
                T.ascend_wait_flag("MTE2_V", eid)
                T.ascend_wait_flag("MTE3_V", eid)
                with T.SimtVF(threads=threads):
                    # Fragment: vectorized float4 load from UB to registers
                    x_frag = T.alloc_fragment((TILE,), "float32")
                    for i in T.Parallel(TILE):
                        x_frag[i] = x_ub[eid, i]

                    # Reduce: sum(x^2) from registers (no UB load)
                    sum_sq = T.alloc_reducer((1,), "float32", op="sum", replication="all")
                    T.clear(sum_sq)
                    for i in T.Parallel(TILE):
                        if i < d:
                            sum_sq[0] += x_frag[i] * x_frag[i]
                    T.finalize_reducer(sum_sq)

                    # Compute rstd
                    var = sum_sq[0] / d + eps
                    rstd_val = T.rsqrt(var)
                    z_rstd_ub[eid, 0] = rstd_val

                    # Output: y = x * rstd * w (reuse x_frag, no x reload)
                    for i in T.Parallel(TILE):
                        if i < d:
                            y_ub[eid, i] = x_frag[i] * rstd_val * w_ub[i]

                T.ascend_set_flag("V_MTE3", eid)
                T.ascend_set_flag("V_MTE2", eid)

                # MTE3: store y[token] and rstd from UB to GM
                row_id = t * N_CORES + core_id
                T.ascend_wait_flag("V_MTE3", eid)
                T.copy(y_ub[eid, :d], Y[base : base + d])
                T.copy(z_rstd_ub[eid, :1], RSTD[row_id : row_id + 1])
                T.ascend_set_flag("MTE3_V", eid)

            # Drain pipeline
            T.ascend_wait_flag("MTE3_V", 0)
            T.ascend_wait_flag("MTE3_V", 1)
            T.ascend_wait_flag("V_MTE2", 0)
            T.ascend_wait_flag("V_MTE2", 1)

    return main
```

---

## 2. 当前 codegen 链路：从 TIR 到 Ascend C

### 2.1 最终 device TIR

经过 `LowerAndLegalize + OptimizeForTarget` 之后，得到下面这份 device TIR，后续 PTO 后端对接都基于它展开。

```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main_kernel(RSTD: T.handle("float32", "global"), W: T.handle("float32", "global"), X: T.handle("float32", "global"), Y: T.handle("float32", "global"), eps: T.float32):
        T.func_attr({"calling_conv": 2, "dyn_shared_memory_buf": 82496, "target": T.target({"keys": ["ascend", "cpu"], "kind": "c", "tag": ""}), "thread_extent": {"blockIdx.x": 64}, "tir.is_global_func": T.bool(True), "tir.kernel_launch_params": ["blockIdx.x", "threadIdx.x", "threadIdx.y", "threadIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tir.use_dyn_shared_memory"], "tir.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [1, 2]})
        buf_dyn_shmem = T.handle("uint8", "shared.dyn")
        w_ub = T.decl_buffer((4096,), data=buf_dyn_shmem, scope="shared.dyn")
        y_ub = T.decl_buffer((8192,), data=buf_dyn_shmem, scope="shared.dyn")
        z_rstd_ub = T.decl_buffer((16,), data=buf_dyn_shmem, scope="shared.dyn")
        sum_sq = T.handle("float32", "local")
        sum_sq_1 = T.decl_buffer((1,), data=sum_sq, scope="local")
        x_ub = T.decl_buffer((8192,), data=buf_dyn_shmem, scope="shared.dyn")
        x_frag = T.handle("float32", "local")
        x_frag_1 = T.decl_buffer((32,), data=x_frag, scope="local")
        bx = T.launch_thread("blockIdx.x", 64)
        buf_dyn_shmem = T.allocate([82496], "uint8", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        T.ascend_copy_gm_to_ubuf(T.tvm_access_ptr(T.type_annotation("float32"), buf_dyn_shmem, 0, 4096, 2), T.tvm_access_ptr(T.type_annotation("float32"), W, 0, 4096, 1), 0, 1, 512, 0, 0)
        T.ascend_set_flag("MTE2_V", 3)
        T.ascend_wait_flag("MTE2_V", 3)
        T.ascend_set_flag("V_MTE2", 0)
        T.ascend_set_flag("MTE3_V", 0)
        T.ascend_set_flag("V_MTE2", 1)
        T.ascend_set_flag("MTE3_V", 1)
        for t in range(64):
            T.ascend_wait_flag("V_MTE2", t % 2)
            T.ascend_copy_gm_to_ubuf(T.tvm_access_ptr(T.type_annotation("float32"), buf_dyn_shmem, t % 2 * 4096 + 4096, 4096, 2), T.tvm_access_ptr(T.type_annotation("float32"), X, t * 262144 + bx * 4096, 4096, 1), 0, 1, 512, 0, 0)
            T.ascend_set_flag("MTE2_V", t % 2)
            T.ascend_wait_flag("MTE2_V", t % 2)
            T.ascend_wait_flag("MTE3_V", t % 2)
            with T.block("SIMT_VF", no_realize=True):
                T.reads()
                T.writes()
                x_frag_2 = T.Buffer((32,), data=x_frag, scope="local")
                sum_sq_2 = T.Buffer((1,), data=sum_sq, scope="local")
                T.block_attr({"layout_map": {x_frag_2: metadata["tl.Fragment"][0], sum_sq_2: metadata["tl.Fragment"][1]}})
                simtvf_tx = T.launch_thread("threadIdx.x", 128)
                simtvf_ty = T.launch_thread("threadIdx.y", 1)
                simtvf_tz = T.launch_thread("threadIdx.z", 1)
                T.attr("simtvf", "tl.simtvf_scope", 1)
                x_frag = T.allocate([32], "float32", "local")
                sum_sq = T.allocate([1], "float32", "local")
                x_frag_3 = T.Buffer((32,), data=x_frag, scope="local")
                for i in T.unroll(8):
                    x_ub_1 = T.Buffer((8192,), data=buf_dyn_shmem, scope="shared.dyn")
                    x_frag_3[i * 4:i * 4 + 4] = x_ub_1[t % 2 * 4096 + i * 512 + simtvf_tx * 4 + 4096:t % 2 * 4096 + i * 512 + simtvf_tx * 4 + 4096 + 4]
                sum_sq_3 = T.Buffer((1,), data=sum_sq, scope="local")
                sum_sq_3[0] = T.float32(0.0)
                for i in T.unroll(32):
                    sum_sq_3[0] = sum_sq_3[0] + x_frag_3[i] * x_frag_3[i]
                sum_sq_3[0] = T.call_extern("float32", "tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run", sum_sq_3[0], T.tvm_access_ptr(T.type_annotation("float32"), buf_dyn_shmem, 20480, 128, 2))
                var: T.float32 = sum_sq_3[0] / T.float32(4096.0) + eps
                rstd_val: T.float32 = T.rsqrt(var)
                z_rstd_ub_1 = T.Buffer((16,), data=buf_dyn_shmem, scope="shared.dyn")
                z_rstd_ub_1[t % 2 * 8 + 20608] = rstd_val
                for i in T.unroll(8):
                    y_ub_1 = T.Buffer((8192,), data=buf_dyn_shmem, scope="shared.dyn")
                    w_ub_1 = T.Buffer((4096,), data=buf_dyn_shmem, scope="shared.dyn")
                    y_ub_1[t % 2 * 4096 + i * 512 + simtvf_tx * 4 + 12288:t % 2 * 4096 + i * 512 + simtvf_tx * 4 + 12288 + 4] = x_frag_3[i * 4:i * 4 + 4] * T.Broadcast(rstd_val, 4) * w_ub_1[i * 512 + simtvf_tx * 4:i * 512 + simtvf_tx * 4 + 4]
            T.ascend_set_flag("V_MTE3", t % 2)
            T.ascend_set_flag("V_MTE2", t % 2)
            T.ascend_wait_flag("V_MTE3", t % 2)
            T.ascend_copy_ubuf_to_gm(T.tvm_access_ptr(T.type_annotation("float32"), Y, t * 262144 + bx * 4096, 4096, 2), T.tvm_access_ptr(T.type_annotation("float32"), buf_dyn_shmem, t % 2 * 4096 + 12288, 4096, 1), 0, 1, 16384, 0, 16384, 16384)
            T.ascend_copy_ubuf_to_gm(T.tvm_access_ptr(T.type_annotation("float32"), RSTD, t * 64 + bx, 1, 2), T.tvm_access_ptr(T.type_annotation("float32"), buf_dyn_shmem, t % 2 * 8 + 20608, 1, 1), 0, 1, 4, 0, 4, 4)
            T.ascend_set_flag("MTE3_V", t % 2)
        T.ascend_wait_flag("MTE3_V", 0)
        T.ascend_wait_flag("MTE3_V", 1)
        T.ascend_wait_flag("V_MTE2", 0)
        T.ascend_wait_flag("V_MTE2", 1)

# Metadata omitted. Use show_meta=True in script() method to show it.
```

后续对接最关键的是其中两处结构。

一处是访存循环：

```python
x_frag_3[i * 4:i * 4 + 4] = x_ub_1[... : ... + 4]
```

另一处是规约调用：

```python
sum_sq_3[0] = T.call_extern(
    "float32",
    "tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run",
    sum_sq_3[0],
    T.tvm_access_ptr(T.type_annotation("float32"), buf_dyn_shmem, 20480, 128, 2),
)
```

### 2.2 当前生成的 Ascend C 代码

```cpp
#include <limits>
__simt_vf__ __launch_bounds__(128) inline void simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t t, float eps) {
  int32_t simtvf_tx = threadIdx.x;
  int32_t simtvf_ty = threadIdx.y;
  int32_t simtvf_tz = threadIdx.z;
  float x_frag[32];
  float sum_sq[1];
  #pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    *(float4*)(x_frag + (i * 4)) = *(__ubuf__ float4*)(((__ubuf__ float*)buf_dyn_shmem) + (((((t & 1) * 4096) + (i * 512)) + (simtvf_tx * 4)) + 4096));
  }
  sum_sq[0] = float(0x0p+0f/*0.000000e+00*/);
  #pragma unroll
  for (int32_t i_1 = 0; i_1 < 32; ++i_1) {
    sum_sq[0] = (sum_sq[0] + (x_frag[i_1] * x_frag[i_1]));
  }
  sum_sq[0] = tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run(sum_sq[0], (&(((__ubuf__ float*)buf_dyn_shmem)[20480])));
  float var = ((sum_sq[0] / float(0x1p+12f/*4.096000e+03*/)) + eps);
  float rstd_val = rsqrtf(var);
  ((__ubuf__ float*)buf_dyn_shmem)[(((t & 1) * 8) + 20608)] = rstd_val;
  #pragma unroll
  for (int32_t i_2 = 0; i_2 < 8; ++i_2) {
    *(__ubuf__ float4*)(((__ubuf__ float*)buf_dyn_shmem) + (((((t & 1) * 4096) + (i_2 * 512)) + (simtvf_tx * 4)) + 12288)) = ((*(float4*)(x_frag + (i_2 * 4)) * make_float4(rstd_val, rstd_val, rstd_val, rstd_val)) * *(__ubuf__ float4*)(((__ubuf__ float*)buf_dyn_shmem) + ((i_2 * 512) + (simtvf_tx * 4))));
  }
}

extern "C"
__global__ __vector__ void main_kernel(__gm__ float* RSTD, __gm__ float* W, __gm__ float* X, __gm__ float* Y, float eps) {
  int32_t bx = blockIdx.x;
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
  copy_gm_to_ubuf((__ubuf__ void*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ void*)((&(W[0]))), 0, 1, 512, 0, 0);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
  for (int32_t t = 0; t < 64; ++t) {
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((t & 1));
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    copy_gm_to_ubuf((__ubuf__ void*)((&(((__ubuf__ float*)buf_dyn_shmem)[(((t & 1) * 4096) + 4096)]))), (__gm__ void*)((&(X[((t * 262144) + (bx * 4096))]))), 0, 1, 512, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((t & 1));
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((t & 1));
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((t & 1));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((t & 1));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((t & 1));
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((t & 1));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((t & 1));
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    asc_vf_call<simt_vf_0>(cce::dim3(128), buf_dyn_shmem, t, eps);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    bisheng::cce::copy_ubuf_to_gm_align_v2((__gm__ void*)((&(Y[((t * 262144) + (bx * 4096))]))), (__ubuf__ void*)((&(((__ubuf__ float*)buf_dyn_shmem)[(((t & 1) * 4096) + 12288)]))), 0, 1, 16384, 0, 16384, 16384);
    bisheng::cce::copy_ubuf_to_gm_align_v2((__gm__ void*)((&(RSTD[((t * 64) + bx)]))), (__ubuf__ void*)((&(((__ubuf__ float*)buf_dyn_shmem)[(((t & 1) * 8) + 20608)]))), 0, 1, 4, 0, 4, 4);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
  }
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}
```

这里有两处和后续讨论直接相关：第一个循环已经是 `float4` 粒度的向量访存；reduce 调用已经实例化为 `tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run(...)`。

---

## 3. PTO 后端对接方案

### 3.1 总体结构

PTO 后端按两层组织。外层是 `pto.aicore` kernel，负责 GM/UB 搬运、`set_flag/wait_flag/barrier`、persistent `for t in range(64)` 循环，并发起 SIMT body 执行。内层是 `pto.simt_entry` body，承担 lane-local 向量访存、32 元素累加、跨 workitem 规约和最终 pointwise 输出。

结构化 SIMT launch 接口在 `feature-simt-ops` 分支中已有，直接用即可：

```mlir
module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @simt_scalar_core_kernel(%arg0: !pto.ptr<i32, gm>) attributes {pto.aicore} {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %dim_z = arith.constant 1 : i32
    %dim_y = arith.constant 1 : i32
    %dim_x = arith.constant 32 : i32

    %ub_out = pto.castptr %c0_i64 : i64 -> !pto.ptr<i32, ub>

    pto.simt_launch @simt_scalar_core_body<<<%dim_x, %dim_y, %dim_z>>>(%ub_out) : (!pto.ptr<i32, ub>) -> ()

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %ub_out, %arg0, %c128_i64
      nburst(%c32_i64, %c128_i64, %c128_i64)
      : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
```

### 3.2 基础映射

基于最终 TIR，基础映射如下：

| TileLang 最终 TIR 构造 | PTO 后端目标表示 | 说明 |
| --- | --- | --- |
| 标量常量 | `arith.constant` | 统一走 `arith` |
| 标量加减乘除、比较、位运算 | `arith.*` | 统一走 `arith` |
| 循环 | `scf.for` | 静态 trip count 保留在 SCF |
| 条件分支 | `scf.if` | 结构化控制流 |
| GM / UB 指针与地址计算 | `!pto.ptr<..., gm/ub>` + `pto.castptr` / `pto.addptr` | 显式 PTO pointer |
| 标量 load/store | `pto.load` / `pto.store` | PTO micro instructions 标量访存 |
| DMA / flag / wait / barrier | 现有 PTO op | 直接复用 PTO 现有基础设施 |
| SIMT body 入口 | `pto.simt_launch` + `pto.simt_entry` | 使用当前 `feature-simt-ops` 的结构化入口 |
| workitem 同步 | `pto.syncthreads` | lowering 到 `llvm.hivm.sync.workitems` |

### 3.3 向量访存对接方案

#### 3.3.1 Ascend 当前路径

第一个循环目前长这样：

```cpp
for (int32_t i = 0; i < 8; ++i) {
  *(float4*)(x_frag + (i * 4)) = *(__ubuf__ float4*)(((__ubuf__ float*)buf_dyn_shmem) + (((((t & 1) * 4096) + (i * 512)) + (simtvf_tx * 4)) + 4096));
}
```

unroll 之后，LLVM IR 已经落到 `<4 x float>` load/store：

```llvm
%4 = load <4 x float>, ptr addrspace(6) %add.ptr, align 16
store <4 x float> %4, ptr %x_frag, align 16

%5 = load <4 x float>, ptr addrspace(6) %add.ptr.1, align 16
store <4 x float> %5, ptr %add.ptr9.1, align 16

%6 = load <4 x float>, ptr addrspace(6) %add.ptr.2, align 16
store <4 x float> %6, ptr %add.ptr9.2, align 16
```

SROA 之后，第二个循环里的 `x_frag[i]` 被改写成对这些向量的 `extractelement`：

```llvm
%4 = load <4 x float>, ptr addrspace(6) %add.ptr, align 16
...
%x_frag.sroa.0.0.vec.extract = extractelement <4 x float> %4, i32 0
%12 = tail call float @llvm.fmuladd.f32(float %x_frag.sroa.0.0.vec.extract, float %x_frag.sroa.0.0.vec.extract, float 0.000000e+00)
%x_frag.sroa.0.4.vec.extract = extractelement <4 x float> %4, i32 1
...
```

可见 Ascend 现有优化链需要的上游形态是：第一层保留 128-bit 向量访存，第二层允许按标量消费，剩下交给 SROA 和 LLVM 把本地数组拆成 `extractelement`。

#### 3.3.2 对接做法

第一个循环直接映射到 LLVM Dialect 的向量 load/store，`x_frag[32]` 保留为 lane-local 局部数组：

- 第一层 `8 x float4` 访存在生成的 SIMT helper 里维持 `<4 x float>` 粒度；
- `x_frag[32]` 作为 lane-local scratch；
- 第二层标量累加照常消费 `x_frag[i]`；
- 由 SROA 和后续 LLVM pass 把 `x_frag` 拆成 `extractelement`。

helper 内 IR 形态大致如下：

```mlir
%xfrag = llvm.alloca ... : !llvm.ptr
%slot0 = llvm.getelementptr %xfrag[%c0] : (!llvm.ptr, i64) -> !llvm.ptr
%v0 = llvm.load %src0 : !llvm.ptr<6> -> vector<4xf32>
llvm.store %v0, %slot0 : !llvm.ptr
```

#### 3.3.3 小结

向量访存这部分的结论：第一层 `8 x float4` 走 LLVM Dialect 的向量 load/store，`x_frag[32]` 作 lane-local scratch 由第二层标量消费，剩下交给 SROA / LLVM 拆开。

如果以后要对接 PTODSL，可以考虑在 PTODSL 中用类似 Python slice 的语法描述 4-lane 连续访存，让局部临时与连续 4 元素窗口保持结构化切片关系，再在 PTODSL lowering 中翻成 LLVM Dialect 的向量 load/store。

### 3.4 Reduce 对接方案

#### 3.4.1 `tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run(...)` 的语义

TIR 中的 reduce 调用是：

```python
sum_sq_3[0] = T.call_extern(
    "float32",
    "tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run",
    sum_sq_3[0],
    T.tvm_access_ptr(T.type_annotation("float32"), buf_dyn_shmem, 20480, 128, 2),
)
```

输入是每个 workitem 的局部标量 `x`，对 128 个 workitem 做一次 sum allreduce，所有参与的 workitem 都拿到同一个结果。四个模板参数依次是 `Reducer=tl::SumOp`、`threads=128`、`scale=1`、`thread_offset=0`。

放在 RMSNorm 的语境里：每个 workitem 先算自己 32 个元素的 `sum(x^2)`，128 个 workitem 共同规约出整行总和，然后各自用这个总和算出同一个 `rstd`。

#### 3.4.2 `AscendAllReduce` 现有实现

`reduce.h` 中相关代码：

```cpp
namespace tl {
struct SumOp {
  template <typename T> __simt_callee__ T operator()(T x, T y) { return x + y; }
};

template <class Reducer> static __simt_callee__ float hw_reduce(float v) {
  if constexpr (std::is_same_v<Reducer, SumOp>)
    return asc_reduce_add(v);
  else if constexpr (std::is_same_v<Reducer, MaxOp>)
    return asc_reduce_max(v);
  else
    return asc_reduce_min(v);
}

template <class Reducer, int threads, int scale = 1, int thread_offset = 0>
struct AscendAllReduce {
  template <class, int, int, int> friend struct AscendAllReduce;

  static_assert(threads >= 1);
  static_assert(scale >= 1);
  static_assert(threads % scale == 0);

private:
  static constexpr bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }
  static constexpr int extent = threads / scale;

  template <int cur, int stop> static __simt_callee__ float butterfly(float x) {
    if constexpr (cur <= stop) {
      return x;
    } else {
      constexpr int offset = cur / 2;
      Reducer op;
      x = op(x, asc_shfl_xor(x, offset));
      return butterfly<cur / 2, stop>(x);
    }
  }

  static __simt_callee__ float warp_hw_reduce(float x) {
    constexpr float id = reduce_identity<Reducer>();
    constexpr int groups = 32 / threads;

    if constexpr (groups == 1) {
      return hw_reduce<Reducer>(x);
    } else {
      int tx = threadIdx.x - thread_offset;
      int my_group = (tx & 31) / threads;
      float result = x;
#pragma unroll
      for (int g = 0; g < groups; ++g) {
        float v = hw_reduce<Reducer>((my_group == g) ? result : id);
        if (my_group == g)
          result = v;
      }
      return result;
    }
  }

  static __simt_callee__ float warp_reduce(float x) {
    if constexpr (extent <= 1) {
      return x;
    } else if constexpr (extent >= 16 && scale == 1) {
      return warp_hw_reduce(x);
    } else {
      return butterfly<threads, scale>(x);
    }
  }

  static __simt_callee__ float ub_reduce(float x, __ubuf__ float *red_buf) {
    int tx = threadIdx.x - thread_offset;
    int group = tx / threads;
    int lane = tx % threads;
    Reducer op;

    red_buf[tx] = x;
    asc_syncthreads();

    float result = x;
    if (lane % scale == 0) {
      result = red_buf[group * threads + lane];
      for (int i = scale; i < threads; i += scale) {
        int idx = group * threads + (lane % scale) + i;
        result = op(result, red_buf[idx]);
      }
    }
    asc_syncthreads();

    if (lane % scale == 0 && lane / scale == 0) {
      red_buf[group * threads + (lane % scale)] = result;
    }
    asc_syncthreads();
    result = red_buf[group * threads + (tx % scale)];
    asc_syncthreads();
    return result;
  }

  static __simt_callee__ float cross_warp_reduce(float x,
                                                 __ubuf__ float *red_buf) {
    constexpr int num_warps = threads / 32;
    int tx = threadIdx.x - thread_offset;
    int wid = tx >> 5, lid = tx & 31;

    float warp_val =
        AscendAllReduce<Reducer, 32, scale, thread_offset>::warp_reduce(x);

    if (lid < scale)
      red_buf[wid * scale + lid] = warp_val;
    asc_syncthreads();

    float result;
    if constexpr (scale == 1) {
      if (tx < 32) {
        float v = (lid < num_warps) ? red_buf[lid] : reduce_identity<Reducer>();
        result = hw_reduce<Reducer>(v);
      }
    } else if constexpr (scale * num_warps <= 32) {
      if (tx < 32) {
        constexpr int total = scale * num_warps;
        float v = (lid < total) ? red_buf[lid] : reduce_identity<Reducer>();
        result = AscendAllReduce<Reducer, total, scale, 0>::warp_reduce(v);
      }
    } else {
      if (tx < 32) {
        Reducer op;
        result = reduce_identity<Reducer>();
        int my_slot = lid % scale;
        for (int w = 0; w < num_warps; ++w) {
          int idx = w * scale + my_slot;
          result = (lid < scale) ? op(result, red_buf[idx]) : result;
        }
      }
    }
    asc_syncthreads();

    if (tx < scale)
      red_buf[tx] = result;
    asc_syncthreads();
    result = red_buf[tx % scale];
    asc_syncthreads();
    return result;
  }

public:
  static __simt_callee__ float run(float x, __ubuf__ float *red_buf = nullptr) {
    if constexpr (threads <= scale) {
      return x;
    } else if constexpr (threads <= 32 && is_pow2(threads) && is_pow2(scale)) {
      return warp_reduce(x);
    } else if constexpr (threads <= 32) {
      return ub_reduce(x, red_buf);
    } else if constexpr (threads > 32 && is_pow2(threads) && scale <= 32 &&
                         is_pow2(scale)) {
      return cross_warp_reduce(x, red_buf);
    } else {
      return ub_reduce(x, red_buf);
    }
  }
};
} // namespace tl
```

对当前的 `SumOp, 128, 1, 0` 实例，模板会选择 `cross_warp_reduce` 路径（threads=128>32，且 threads 和 scale 都是 2 的幂），执行步骤是：

1. 128 个 workitem 分成 4 个 32-lane 子块；
2. 每个子块各做一次 `asc_reduce_add`；
3. 4 个子块的 leader 把部分和写入 UB scratch；
4. `asc_syncthreads()` 同步；
5. 一个 32-lane 子块再把这 4 个部分和规约成最终结果；
6. 全局 leader 写回 scratch[0]；
7. 再来一次 `asc_syncthreads()`，把最终结果广播给全部 128 个 workitem。

#### 3.4.3 PTO codegen 做法

做法是把"模板实例化"放到 codegen 阶段：根据 `AscendAllReduce` 的模板参数和数据类型，由 TileLang -> PTO lowering 直接生成特化好的 PTO IR helper。这样模板语义就停留在 TileLang 侧，PTO 后端拿到的是普通 PTO IR。

以当前实例（`Reducer=sum, dtype=f32, threads=128, scale=1, thread_offset=0`）为例，生成的 helper 大致是：

```mlir
func.func private @__tl_allreduce_sum_f32_t128_s1_o0(
    %x: f32,
    %scratch: !pto.ptr<f32, ub>
) -> f32 attributes {
  tl.reducer = "sum",
  tl.dtype = "f32",
  tl.threads = 128 : i32,
  tl.scale = 1 : i32,
  tl.thread_offset = 0 : i32
} {
  %laneid = pto.get_laneid : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c32 = arith.constant 32 : i32

  %subgroup = arith.divui %laneid, %c32 : i32
  %lane_in_group = arith.remui %laneid, %c32 : i32

  // stage 1: 32-lane subgroup reduction
  %warp_sum = pto.redux_add %x : f32 -> f32

  // subgroup leader writes partial sum
  %is_leader = arith.cmpi eq, %lane_in_group, %c0 : i32
  scf.if %is_leader {
    %idx = arith.index_castui %subgroup : i32 to index
    pto.store %warp_sum, %scratch[%idx] : !pto.ptr<f32, ub>, f32
  }

  pto.syncthreads

  // stage 2: subgroup 0 reduces 4 partial sums
  %is_group0 = arith.cmpi eq, %subgroup, %c0 : i32
  %need_load = arith.andi %is_group0, arith.cmpi ult, %lane_in_group, %c4 : i32
  %v = scf.if %need_load -> f32 {
    %idx = arith.index_castui %lane_in_group : i32 to index
    %tmp = pto.load %scratch[%idx] : !pto.ptr<f32, ub> -> f32
    scf.yield %tmp : f32
  } else {
    %zero = arith.constant 0.0 : f32
    scf.yield %zero : f32
  }

  %total = scf.if %is_group0 -> f32 {
    %r = pto.redux_add %v : f32 -> f32
    scf.yield %r : f32
  } else {
    %undef = arith.constant 0.0 : f32
    scf.yield %undef : f32
  }

  %is_global_leader = arith.cmpi eq, %laneid, %c0 : i32
  scf.if %is_global_leader {
    %idx0 = arith.index_castui %c0 : i32 to index
    pto.store %total, %scratch[%idx0] : !pto.ptr<f32, ub>, f32
  }

  pto.syncthreads

  %idx0 = arith.index_castui %c0 : i32 to index
  %result = pto.load %scratch[%idx0] : !pto.ptr<f32, ub> -> f32
  return %result : f32
}
```

原 TIR 的 reduce 调用：

```python
sum_sq_3[0] = T.call_extern(
    "float32",
    "tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run",
    sum_sq_3[0],
    T.tvm_access_ptr(T.type_annotation("float32"), buf_dyn_shmem, 20480, 128, 2),
)
```

在 PTO lowering 中改写为对 helper 的调用：

```mlir
%scratch = ... : !pto.ptr<f32, ub>
%sum = func.call @__tl_allreduce_sum_f32_t128_s1_o0(%partial, %scratch)
  : (f32, !pto.ptr<f32, ub>) -> f32
```

要点：

- helper 按实例自动生成，`<threads, scale, offset>` 的不同组合各自对应一份；
- `threads/scale/thread_offset` 都是编译期常量，每个实例落到独立 helper；
- PTO 后端看到的是普通 PTO IR；
- helper 上的 attrs 只用于调试展示，后端不会再读 attrs 做二次特化。

后续如果接入 PTODSL，可以在 PTODSL 中把 Reduce 函数写好，然后在 codegen 阶段按 `AscendAllReduce` 的实例参数选择性 import 并调用。collective 语义就提前沉淀到 PTODSL，codegen 仍按实例选具体实现。

### 3.5 TIR -> PTO 映射表

| 最终 TIR 构造 | PTO 后端目标表示 | 说明 |
| --- | --- | --- |
| `main_kernel` | `func.func @main_kernel ... attributes {pto.aicore}` | 外层 AICORE kernel |
| `SIMT_VF` block | `pto.simt_launch @body<<<dim_x, dim_y, dim_z>>>(...)` + `func.func @body ... attributes {pto.simt_entry}` | 使用结构化 SIMT launch |
| `simtvf_tx` / `threadIdx.x` | `pto.get_tid_x` | SIMT body 内获取 workitem 坐标 |
| 标量常量、算术、比较 | `arith.*` | 基础标量 IR |
| `for` / `if` | `scf.for` / `scf.if` | 基础结构化控制流 |
| 第一个 `8 x float4` 访存循环 | LLVM Dialect 向量 load/store | 保留 128-bit bundle 信息 |
| `x_frag[32]` | lane-local scratch，允许后续 SROA 消除 | 第一版对齐当前 Ascend 路径 |
| `tl::AscendAllReduce<...>::run` | 自动实例化的特化 PTO IR helper + `func.call` | helper 内部使用 `pto.redux_add + pto.syncthreads + pto.load/store` |
| `rsqrt` | `sqrt + reciprocal/div` | 当前无需先扩展 PTO 标量 op 面 |
| GM/UB 搬运与 flag | `pto.mte_*` / `pto.set_flag` / `pto.wait_flag` / `pto.barrier` | 外层 kernel 直接使用已有 PTO op |

---

## 4. 主要实现工作

`tilelang` 中 PTO 后端的基础已经存在，新增工作集中在下面几个方向。

### 4.1 `SIMT_VF` 结构化外提

在最终 device TIR 上识别 `SIMT_VF` block，并生成：

- 外层 `pto.aicore` kernel；
- 内层 `pto.simt_entry` body；
- 两者之间的 `pto.simt_launch`；
- 必要时补 `pto.simt_max_threads` 等 launch config。

### 4.2 `float4` 向量访存 lowering

需要做的事：

1. 识别 `BufferLoad/BufferStore` 上 `Ramp(base, 1, 4)` 这种连续 4-lane 访存；
2. 第一层 `8 x float4` 直接映射到 LLVM Dialect 向量 load/store；
3. 保住 `x_frag` 的 lane-local scratch 形态，第二层标量消费照常，让 SROA 拆成 `extractelement`；
4. 保证这一路结果能继续走 SROA / LLVM 优化链。

### 4.3 `AscendAllReduce` helper 自动实例化

引入一个内部实例签名，包含 reducer kind、dtype、threads、scale、thread_offset。基于这个签名：

1. 自动生成 helper 名字，例如 `__tl_allreduce_sum_f32_t128_s1_o0`、`__tl_allreduce_sum_f32_t32_s1_o0`、`__tl_allreduce_max_f32_t128_s1_o0`；
2. 在 module 内做 helper cache，同一实例只生成一次；
3. 把 TIR 中 `tl::AscendAllReduce<...>::run` 的调用点改写为对 helper 的 `func.call`。

### 4.4 helper body 自动生成

helper body 的生成逻辑需要覆盖：

- `threads <= 32`：单级 `pto.redux_*`；
- `threads > 32`：两级规约，包含 subgroup leader 写部分和、`pto.syncthreads`、第二级规约、global leader 写回、`pto.syncthreads`、广播读；
- `scale > 1`：生成 slot / subgroup 映射逻辑；
- `SumOp / MaxOp / MinOp` 共用同一个 generator，只换 reducer。

### 4.5 标量 math 与 local scratch

剩下的零散工作：

- `rsqrt` 按 `sqrt + reciprocal/div` 展开；
- lane-local 标量 scratch 和临时数组的表示与 lowering；
- SIMT body 内 `pto.syncthreads` 的插入位置和验证。

### 4.6 测试与验证

测试分三层：

1. **IR 级**：检查 `pto.simt_launch`、`pto.syncthreads`、自动生成的 `__tl_allreduce_*` helper，以及 helper 中的 `pto.redux_add`；
2. **LLVM 级**：检查 `llvm.hivm.sync.workitems`、`<4 x float>` load/store，以及后续的 `extractelement`；
3. **功能与性能级**：先以 RMSNorm 作为第一条 vertical slice，再扩展到其它 `SimtVF` kernel。

---

## 结论

对当前 TileLang RMSNorm kernel 而言，对接 PTO 后端的要点是：以最终 device TIR 为输入；`SIMT_VF` 映射到 `pto.simt_launch + pto.simt_entry`；第一层 `float4` 访存保留为 MLIR/LLVM 128-bit 向量 load/store；`tl::AscendAllReduce<...>::run` 用自动实例化的特化 PTO IR helper 承接。

`feature-simt-ops` 上已经具备 `pto.syncthreads -> llvm.hivm.sync.workitems`，因此两级 allreduce 的 PTO IR 表达所需的关键同步原语已经齐了。

整体方向是：TileLang 侧把高层语义展开清楚；PTO 后端只编译普通 PTO IR 加必要的 LLVM 向量访存；尽量复用 Ascend 路径已经验证过的 `<4 x float>` + SROA 优化链。
