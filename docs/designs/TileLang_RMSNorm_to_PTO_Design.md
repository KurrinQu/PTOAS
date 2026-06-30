# TileLang RMSNorm 对接 PTO 后端方案设计

## 1. TileLang RMSNorm Kernel

讨论 PTO 后端对接之前，先把作为输入的 TileLang RMSNorm kernel 摆出来。它的结构有几个要点：

- 外层 `T.Kernel(N_CORES)` 表示 64 个 AICORE block；
- `T.Pipelined(n_tokens_per_core, num_stages=2)` 表示每个 core 处理 64 个 token，并启用 2-stage 软件流水；
- 前端声明单维 `x_ub / y_ub / z_rstd_ub`，双缓冲 offset、MTE/V/MTE3 flag 同步由 Ascend lowering pipeline 生成；
- 每个 token 的主体计算放进 `T.SimtVF(threads=threads)` 内，`x_frag` 是 SIMT workitem 的本地 fragment；
- 跨 workitem 的规约由 `T.alloc_reducer + T.finalize_reducer` 完成；
- `d=4096`、`threads=128` 下，每个 workitem 负责 32 个 f32 元素，Ascend 路径生成 16 组 `float2` 连续访存。

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
            x_ub = T.alloc_shared((TILE), "float32")
            y_ub = T.alloc_shared((TILE), "float32")
            z_rstd_ub = T.alloc_shared((8), "float32")

            # Load weights once
            T.copy(W[:d], w_ub[:d])

            for t in T.Pipelined(n_tokens_per_core, num_stages=2):
                base = (t * N_CORES + core_id) * d

                T.copy(X[base : base + d], x_ub[:d])

                with T.SimtVF(threads=threads):
                    # Fragment: vectorized float4 load from UB to registers
                    x_frag = T.alloc_fragment((TILE,), "float32")
                    for i in T.Parallel(TILE):
                        x_frag[i] = x_ub[i]

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
                    z_rstd_ub[0] = rstd_val

                    # Output: y = x * rstd * w (reuse x_frag, no x reload)
                    for i in T.Parallel(TILE):
                        if i < d:
                            y_ub[i] = x_frag[i] * rstd_val * w_ub[i]

                # MTE3: store y[token] and rstd from UB to GM
                row_id = t * N_CORES + core_id
                T.copy(y_ub[:d], Y[base : base + d])
                T.copy(z_rstd_ub[:1], RSTD[row_id : row_id + 1])

    return main
```

---

## 2. codegen 链路：从 TIR 到 Ascend C

Ascend 后端通过 `tilelang.backend.pass_pipeline` 注册 lowering pipeline。`tilelang/ascend/pipeline.py` 中的实现如下：

```python
from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target
from tvm.tirx import PrimFunc, SBlock
from tvm.tirx.stmt_functor import post_order_visit

import tilelang
from tilelang.backend.pass_pipeline import PassPipeline, register_pipeline
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    allow_autoschedule,
    allow_global_thread_synchronization,
    allow_vectorize,
    should_disable_shared_memory_reuse,
    should_enable_race_check,
    should_force_let_inline,
)

from . import transform as ascend_transform


def _has_vf_region(mod: IRModule, names: set[str] | None = None) -> bool:
    names = names or {"SIMD_VF", "SIMT_VF"}
    for _, func in mod.functions.items():
        if not isinstance(func, PrimFunc):
            continue
        found = False

        def _visit(node):
            nonlocal found
            if found:
                return
            if isinstance(node, SBlock) and node.name_hint in names:
                found = True

        post_order_visit(func.body, _visit)
        if found:
            return True
    return False


def AscendPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    mod = tirx.transform.BindTarget(target)(mod)
    # Materialize the target-neutral kernel-launch nest (thread_binding For
    # loops emitted by T.Kernel) into thread_extent AttrStmts. Ascend's NPU
    # launch is a 1-D blockIdx.x grid with no threadIdx, so SIMT-style
    # materialization (lower_thread_binding=True) reproduces the previous
    # LaunchThread(blockIdx.x) behavior.
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = ascend_transform.HoistSimdPairs()(mod)
    pass_ctx = tilelang.transform.get_pass_context()

    if should_force_let_inline(pass_ctx=pass_ctx):
        mod = tilelang.transform.LetInline()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    if should_enable_race_check(pass_ctx=pass_ctx):
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    tilelang.ascend.analysis.VFChecker()(mod)

    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    LayoutVisual(mod)

    if allow_autoschedule(pass_ctx=pass_ctx):
        mod = ascend_transform.AnnotateMultiBufferEligible()(mod)
        mod = ascend_transform.IfConditionExtract()(mod)
        mod = ascend_transform.AutoSchedule()(mod)
        mod = ascend_transform.NormalizeMixedKernelSid()(mod)
        mod = tilelang.transform.Simplify()(mod)

    mod = ascend_transform.AscendSimdVFLowerParallel()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)

    mod = tilelang.transform.DecoupleTypeCast()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.ascend.analysis.VFLocalVarChecker()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)

    if not _has_vf_region(mod):
        mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    mod = s_tir.transform.HoistIfThenElse()(mod)

    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    mod = s_tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)

    if allow_global_thread_synchronization(pass_ctx=pass_ctx):
        mod = tilelang.transform.ThreadSync("global")(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = ascend_transform.MarkScalarDcacheBypass()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    disable_reuse = should_disable_shared_memory_reuse(pass_ctx=pass_ctx)
    mod = ascend_transform.MergeUBAllocations(
        enable_aggressive_merge=False,
        align_bytes=32,
        disable_reuse=disable_reuse,
    )(mod)

    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod


ascend_pipeline = PassPipeline("ascend", AscendPassPipelineBody)

register_pipeline(ascend_pipeline)
```

对 RMSNorm 这个 kernel，关键 pass 的作用如下：

- `MaterializeKernelLaunch` 把 `T.Kernel` 产生的 target-neutral launch loop 物化成 Ascend 的 `blockIdx.x` thread extent；
- `AutoSchedule` 相关 pass 根据 `T.Pipelined(num_stages=2)` 生成双缓冲 offset 和 `V_MTE2 / MTE2_V / V_MTE3 / MTE3_V` 同步；
- `AscendSimdVFLowerParallel` 把 `T.Parallel(TILE)` 降到 SIMT workitem 内的固定步长循环；
- `LowerTileOp` 把 `T.copy` 继续 lower 成 Ascend copy intrinsic；
- `MergeUBAllocations` 把多个 `shared.dyn` buffer 合成一个 `buf_dyn_shmem` backing store，并决定最终 offset；
- `LowerThreadAllreduce` 之后，TIR 中仍保留 `tl::AscendAllReduce<...>::run` 外部调用，后续 Ascend C / Bisheng 路径负责模板实例化。

### 2.1 最终 device TIR

以下代码来自 `batch=4096, d=4096` 的 Ascend lowering 结果，对应 `analysis_outputs/rmsnorm_d4096_tir_passes/062_tirx.Filter.py`。

```python
# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main_kernel(RSTD: T.handle("float32", "global"), W: T.handle("float32", "global"), X: T.handle("float32", "global"), Y: T.handle("float32", "global"), eps: T.float32):
        T.func_attr({"calling_conv": 2, "dyn_shared_memory_buf": 82496, "target": T.target({"keys": ["ascend"], "kind": "c", "tag": "", "target_device_type": 12}), "thread_extent": {"blockIdx.x": 64}, "tirx.is_global_func": T.bool(True), "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x", "threadIdx.y", "threadIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tirx.use_dyn_shared_memory"], "tirx.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [1, 2]})
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
        buf_dyn_shmem_1 = T.alloc_buffer((82496,), "uint8", scope="shared.dyn")
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        T.ascend_set_flag("V_MTE2", 0)
        T.ascend_set_flag("V_MTE2", 1)
        T.ascend_set_flag("MTE3_V", 0)
        T.ascend_set_flag("MTE3_V", 1)
        T.ascend_copy_gm_to_ubuf(T.tvm_access_ptr(T.type_annotation("float32"), buf_dyn_shmem, 0, 4096, 2), T.tvm_access_ptr(T.type_annotation("float32"), W, 0, 4096, 1), 0, 1, 16384, 0, 0, 0, 0, 16384, 16384)
        T.ascend_set_flag("MTE2_V", 2)
        for t in range(64):
            T.ascend_wait_flag("V_MTE2", t % 2)
            T.ascend_copy_gm_to_ubuf(T.tvm_access_ptr(T.type_annotation("float32"), buf_dyn_shmem, t % 2 * 4096 + 4224, 4096, 2), T.tvm_access_ptr(T.type_annotation("float32"), X, t * 262144 + bx * 4096, 4096, 1), 0, 1, 16384, 0, 0, 0, 0, 16384, 16384)
            T.ascend_set_flag("MTE2_V", t % 2)
            if t == 0:
                T.ascend_wait_flag("MTE2_V", 2)
            T.ascend_wait_flag("MTE3_V", t % 2)
            T.ascend_wait_flag("MTE2_V", t % 2)
            with T.sblock("SIMT_VF", no_realize=True):
                T.reads()
                T.writes()
                x_frag_2 = T.Buffer((32,), data=x_frag, scope="local")
                sum_sq_2 = T.Buffer((1,), data=sum_sq, scope="local")
                T.sblock_attr({"layout_map": {x_frag_2: metadata["tl.Fragment"][0], sum_sq_2: metadata["tl.Fragment"][1]}, "tl.vf_source_index": T.int64(0)})
                simtvf_tx = T.launch_thread("threadIdx.x", 128)
                simtvf_ty = T.launch_thread("threadIdx.y", 1)
                simtvf_tz = T.launch_thread("threadIdx.z", 1)
                T.attr("simtvf", "tl.simtvf_scope", 1)
                x_frag_3 = T.alloc_buffer((32,), scope="local")
                sum_sq_3 = T.alloc_buffer((1,), scope="local")
                for i in T.unroll(16):
                    x_frag_1[i * 2:i * 2 + 2] = x_ub[t % 2 * 4096 + i * 256 + simtvf_tx * 2 + 4224:t % 2 * 4096 + i * 256 + simtvf_tx * 2 + 4224 + 2]
                sum_sq_1[0] = T.float32(0.0)
                for i in T.unroll(32):
                    sum_sq_1[0] = sum_sq_1[0] + x_frag_1[i] * x_frag_1[i]
                sum_sq_1[0] = T.call_extern("float32", "tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run", sum_sq_1[0], T.tvm_access_ptr(T.type_annotation("float32"), buf_dyn_shmem, 4096, 128, 2))
                var: T.float32 = sum_sq_1[0] / T.float32(4096.0) + eps
                rstd_val: T.float32 = T.rsqrt(var)
                z_rstd_ub[t % 2 * 8 + 20608] = T.rsqrt(var)
                for i in T.unroll(16):
                    y_ub[t % 2 * 4096 + i * 256 + simtvf_tx * 2 + 12416:t % 2 * 4096 + i * 256 + simtvf_tx * 2 + 12416 + 2] = x_frag_1[i * 2:i * 2 + 2] * T.Broadcast(T.rsqrt(var), 2) * w_ub[i * 256 + simtvf_tx * 2:i * 256 + simtvf_tx * 2 + 2]
            T.ascend_set_flag("V_MTE2", t % 2)
            T.ascend_set_flag("V_MTE3", t % 2)
            T.ascend_wait_flag("V_MTE3", t % 2)
            T.ascend_copy_ubuf_to_gm(T.tvm_access_ptr(T.type_annotation("float32"), RSTD, t * 64 + bx, 1, 2), T.tvm_access_ptr(T.type_annotation("float32"), buf_dyn_shmem, t % 2 * 8 + 20608, 1, 1), 0, 1, 4, 4, 4, 4)
            T.ascend_copy_ubuf_to_gm(T.tvm_access_ptr(T.type_annotation("float32"), Y, t * 262144 + bx * 4096, 4096, 2), T.tvm_access_ptr(T.type_annotation("float32"), buf_dyn_shmem, t % 2 * 4096 + 12416, 4096, 1), 0, 1, 16384, 4, 16384, 16384)
            T.ascend_set_flag("MTE3_V", t % 2)
        T.ascend_wait_flag("V_MTE2", 0)
        T.ascend_wait_flag("V_MTE2", 1)
        T.ascend_wait_flag("MTE3_V", 0)
        T.ascend_wait_flag("MTE3_V", 1)

# Metadata omitted. Use show_meta=True in script() method to show it.
```

后续对接最关键的是三类结构：

1. `T.Pipelined` 在 TIR 中展开为显式 double-buffer offset 和 flag 同步；
2. SIMT body 中的连续访存是 `16 x float2`，TIR 表达为长度为 2 的切片 load/store；
3. `T.finalize_reducer` 最终以 `tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run` 表达，scratch 指向 UB backing store 中的 `buf_dyn_shmem[4096]`。

### 2.2 生成的 Ascend C 代码

`d=4096` 生成的 Ascend C 完整代码如下：

```cpp
#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(128) inline void simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t t, float eps) {
  float x_frag[32];
  float sum_sq[1];
  #pragma unroll
  for (int32_t i = 0; i < 16; ++i) {
    *(float2*)(x_frag + (i * 2)) = *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((((t & 1) * 4096) + (i * 256)) + (((int32_t)threadIdx.x) * 2)) + 4224));
  }
  sum_sq[0] = float(0x0p+0f/*0.000000e+00*/);
  #pragma unroll
  for (int32_t i_1 = 0; i_1 < 32; ++i_1) {
    sum_sq[0] = (sum_sq[0] + (x_frag[i_1] * x_frag[i_1]));
  }
  sum_sq[0] = tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run(sum_sq[0], (&(((__ubuf__ float*)buf_dyn_shmem)[4096])));
  float var = ((sum_sq[0] / float(0x1p+12f/*4.096000e+03*/)) + eps);
  float rstd_val = (float(0x1p+0f/*1.000000e+00*/) / sqrtf(var));
  ((__ubuf__ float*)buf_dyn_shmem)[(((t & 1) * 8) + 20608)] = (float(0x1p+0f/*1.000000e+00*/) / sqrtf(var));
  #pragma unroll
  for (int32_t i_2 = 0; i_2 < 16; ++i_2) {
    *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((((t & 1) * 4096) + (i_2 * 256)) + (((int32_t)threadIdx.x) * 2)) + 12416)) = ((*(float2*)(x_frag + (i_2 * 2)) * make_float2((float(0x1p+0f/*1.000000e+00*/) / sqrtf(var)), (float(0x1p+0f/*1.000000e+00*/) / sqrtf(var)))) * *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + ((i_2 * 256) + (((int32_t)threadIdx.x) * 2))));
  }
}

__global__ __vector__ void main_kernel(__gm__ float* RSTD, __gm__ float* W, __gm__ float* X, __gm__ float* Y, float eps) {
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
  copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(W[0]))), 0, 1, 16384, 0, 0, 0, 0, 16384, 16384);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
  for (int32_t t = 0; t < 64; ++t) {
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((t & 1));
    copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[(((t & 1) * 4096) + 4224)]))), (__gm__ uint8_t*)((&(X[((t * 262144) + (((int32_t)blockIdx.x) * 4096))]))), 0, 1, 16384, 0, 0, 0, 0, 16384, 16384);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((t & 1));
    if (t == 0) {
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((t & 1));
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((t & 1));
    asc_vf_call<simt_vf_0>(cce::dim3(128), buf_dyn_shmem, t, eps);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((t & 1));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((t & 1));
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((t & 1));
    copy_ubuf_to_gm_align_v2((__gm__ void*)((&(RSTD[((t * 64) + ((int32_t)blockIdx.x))]))), (__ubuf__ void*)((&(((__ubuf__ float*)buf_dyn_shmem)[(((t & 1) * 8) + 20608)]))), 0, 1, 4, 4, 4, 4);
    copy_ubuf_to_gm_align_v2((__gm__ void*)((&(Y[((t * 262144) + (((int32_t)blockIdx.x) * 4096))]))), (__ubuf__ void*)((&(((__ubuf__ float*)buf_dyn_shmem)[(((t & 1) * 4096) + 12416)]))), 0, 1, 16384, 4, 16384, 16384);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((t & 1));
  }
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
}

#ifdef __cplusplus
extern "C"
#endif
void __tl_launch_main_kernel(void** void_args, uint32_t grid, uint32_t smem, void* stream) {
  (void)grid; (void)smem;
  main_kernel<<<64, 82496, stream>>>((*reinterpret_cast<__gm__ float**>(void_args[0])), (*reinterpret_cast<__gm__ float**>(void_args[1])), (*reinterpret_cast<__gm__ float**>(void_args[2])), (*reinterpret_cast<__gm__ float**>(void_args[3])), (*reinterpret_cast<float*>(void_args[4])));
}
```

这里有几处和 PTO 后端直接相关：

- SIMT helper 的 UB/local 连续访存是 `float2`，对应 LLVM IR 中的 `<2 x float>` load/store；
- `x_frag[32]` 是 workitem-local scratch，reduce 逐元素消费；
- `AscendAllReduce<SumOp, 128, 1, 0>` 的 scratch 地址来自最终 TIR 的调用参数；
- 外层 MTE/flag 同步由 pipeline pass 生成，PTO 后端按最终 TIR lowering。

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
| SIMT body 入口 | `pto.simt_launch` + `pto.simt_entry` | 使用结构化 SIMT 入口 |
| workitem 同步 | `pto.syncthreads` | lowering 到 `llvm.hivm.sync.workitems` |

### 3.3 向量访存对接方案

#### 3.3.1 Ascend 路径形态

`d=4096` 的 SIMT helper 中，第一个循环是 `16 x float2`：

```cpp
#pragma unroll
for (int32_t i = 0; i < 16; ++i) {
  *(float2*)(x_frag + (i * 2)) =
      *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) +
      (((((t & 1) * 4096) + (i * 256)) + (((int32_t)threadIdx.x) * 2)) + 4224));
}
```

输出循环同样是 `float2`：

```cpp
#pragma unroll
for (int32_t i_2 = 0; i_2 < 16; ++i_2) {
  *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) +
      (((((t & 1) * 4096) + (i_2 * 256)) + (((int32_t)threadIdx.x) * 2)) + 12416)) =
      (*(float2*)(x_frag + (i_2 * 2)) * make_float2(rstd_val, rstd_val)) *
      *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) +
      ((i_2 * 256) + (((int32_t)threadIdx.x) * 2)));
}
```

对应到 TIR，是长度为 2 的连续切片：

```python
for i in T.unroll(16):
    x_frag[i * 2:i * 2 + 2] = \
        x_ub[(t & 1) * 4096 + i * 256 + simtvf_tx * 2 + 4224:
             (t & 1) * 4096 + i * 256 + simtvf_tx * 2 + 4224 + 2]

for i in T.unroll(16):
    y_ub[(t & 1) * 4096 + i * 256 + simtvf_tx * 2 + 12416:
         (t & 1) * 4096 + i * 256 + simtvf_tx * 2 + 12416 + 2] = \
        x_frag[i * 2:i * 2 + 2] * T.Broadcast(rstd_val, 2) * \
        w_ub[i * 256 + simtvf_tx * 2:i * 256 + simtvf_tx * 2 + 2]
```

这一路需要保留两个信息：第一层连续访存有明确 bundle 信息，bundle 宽度是 2 个 f32；第二层 reduce 按标量逐元素消费 `x_frag[32]`。

#### 3.3.2 对接做法

PTO 后端按连续 lane bundle 进行 lowering，`d=4096` 对应 `vector<2xf32>`：

- 识别 TIR 中连续 2-lane 的 `BufferLoad/BufferStore`，即 `Ramp(base, 1, 2)` 或等价 slice；
- SIMT helper 内生成 LLVM Dialect 的 `vector<2xf32>` load/store，地址空间保持 UB / local 的区别；
- `x_frag[32]` 作为 lane-local scratch 保留，后续标量循环继续按 `x_frag[i]` 消费；
- 允许 LLVM SROA / mem2reg / vector scalarization 把 local scratch 进一步拆成 `extractelement` 或标量值；
- 对 `d=5120 / d=7168` 这类 shape，SIMT body 中会保留尾部保护条件，向量 store lowering 需要保留这些条件控制流。

helper 内 IR 形态大致如下：

```mlir
%xfrag = llvm.alloca ... : !llvm.ptr
%src = llvm.getelementptr %ub_base[%offset] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>
%dst = llvm.getelementptr %xfrag[%local_offset] : (!llvm.ptr, i64) -> !llvm.ptr
%v = llvm.load %src : !llvm.ptr<6> -> vector<2xf32>
llvm.store %v, %dst : vector<2xf32>, !llvm.ptr
```

这层可以实现成通用连续 lane bundle lowering：`vector<Nxf32>` 的 N 来自 TIR slice/ramp lanes。

#### 3.3.3 小结

向量访存这部分的结论：PTO 后端需要稳定支持 `vector<2xf32>` 形态。`x_frag[32]` 仍作为 local scratch，由 reduce 和 pointwise 两段共同消费。连续访存的 bundle 宽度从 TIR lanes 推导，避免把实现绑定到某个固定 shape。

### 3.4 Reduce 对接方案

#### 3.4.1 `tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run(...)` 的调用形态

TIR 中的 reduce 调用是外部模板函数调用：

```python
sum_sq[0] = T.call_extern(
    "float32",
    "tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run",
    sum_sq[0],
    T.address_of(buf_dyn_shmem[4096]),
)
```

生成的 Ascend C 代码保持同一个模板接口：

```cpp
sum_sq[0] = tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run(
    sum_sq[0], (&(((__ubuf__ float*)buf_dyn_shmem)[4096])));
```

`common.h` 通过 `#include "tl_templates/ascend/reduce.h"` 引入模板实现。接口定义在 `src/tl_templates/ascend/reduce.h`：

```cpp
template <class Reducer, int threads, int scale = 1, int thread_offset = 0>
struct AscendAllReduce {
  template <typename T>
  static __simt_callee__ T run(T x, __ubuf__ T *red_buf = nullptr);
};
```

四个模板参数的含义如下：

- `Reducer`：规约算子，当前模板支持 `SumOp / MaxOp / MinOp`；
- `threads`：参与 reduce 维度的线程位置数，等于 `extent * scale`；
- `scale`：逻辑参与者在 `threadIdx.x` 空间中的步长；
- `thread_offset`：block 内线程索引偏移，内部使用 `threadIdx.x - thread_offset` 得到局部线程号。

RMSNorm 这里使用 `SumOp, 128, 1, 0`，表示对 128 个连续 SIMT workitem 做 sum allreduce。每个 workitem 先计算自己负责的 32 个元素的 `sum(x^2)`，128 个 workitem 共同规约得到整行平方和，然后各自用这个总和计算 `rstd`。

#### 3.4.2 `reduce.h` 中的实现结构

`reduce.h` 里先定义 reducer functor。`SumOp` 的实现是一个二元加法 functor：

```cpp
struct SumOp {
  template <typename T> __simt_callee__ T operator()(T x, T y) { return x + y; }
};
```

寄存器规约路径通过 `HwReduce<Reducer, T>` 特化连接到硬件 reduce intrinsic。`SumOp` 对 `float / int32_t / uint32_t` 都映射到 `asc_reduce_add`：

```cpp
template <class Reducer, typename T, typename = void> struct HwReduce;

template <> struct HwReduce<SumOp, float> {
  static __simt_callee__ float call(float v) { return asc_reduce_add(v); }
};
template <> struct HwReduce<SumOp, int32_t> {
  static __simt_callee__ int32_t call(int32_t v) { return asc_reduce_add(v); }
};
template <> struct HwReduce<SumOp, uint32_t> {
  static __simt_callee__ uint32_t call(uint32_t v) { return asc_reduce_add(v); }
};
```

butterfly 路径通过 `ShflXor<T>` 特化连接到 `asc_shfl_xor`。`float` 的实现如下，`int32_t / uint32_t / int64_t / uint64_t` 也有同形态特化：

```cpp
template <typename T, typename = void> struct ShflXor;

template <> struct ShflXor<float> {
  static __simt_callee__ float call(float v, int o) {
    return asc_shfl_xor(v, o);
  }
};
```

核心实现分成三类：

1. `warp_reduce`：warp 内寄存器规约。优先使用 `HwReduce<Reducer,T>`；当硬件 reduce 不适合时，使用 `asc_shfl_xor` butterfly；
2. `cross_warp_reduce`：`threads > 32` 时先做 warp 内规约，各 warp 把部分结果写入 UB，`asc_syncthreads()` 后由一个 warp 做第二级规约，再把结果写回 UB 并广播给所有线程；
3. `ub_reduce`：通用 UB fallback。所有线程先把输入写入 `red_buf`，同步后按 `scale` 汇总，再同步广播。

`run` 的 dispatch 逻辑如下：

```cpp
template <typename T>
static __simt_callee__ T run(T x, __ubuf__ T *red_buf = nullptr) {
  if constexpr (threads <= scale) {
    return x;
  } else if constexpr (threads <= 32 && is_pow2(threads) && is_pow2(scale)) {
    return warp_or_ub(x, red_buf, 0);
  } else if constexpr (threads <= 32) {
    return ub_reduce(x, red_buf);
  } else if constexpr (threads > 32 && is_pow2(threads) && scale <= 32 &&
                       is_pow2(scale)) {
    return cross_warp_or_ub(x, red_buf, 0);
  } else {
    return ub_reduce(x, red_buf);
  }
}
```

对 `SumOp, 128, 1, 0`：

- `threads > 32`、`threads` 是 2 的幂、`scale=1` 是 2 的幂，进入 `cross_warp_or_ub`；
- `float + SumOp` 有 `HwReduce<SumOp, float>` 特化，所以选择 `cross_warp_reduce`；
- `cross_warp_reduce` 内部先调用 `AscendAllReduce<Reducer, 32, scale, thread_offset>::warp_reduce` 做每个 warp 的部分和；
- `scale == 1` 时，第二级规约使用 `HwReduce<Reducer, U>::call(v)`，也就是 `asc_reduce_add`；
- 期间通过 UB scratch 和多次 `asc_syncthreads()` 保证跨 warp 数据可见和结果广播。

#### 3.4.3 PTO codegen 对接做法

PTO 路径把 `AscendAllReduce` 的模板语义前移到 TileLang -> PTO lowering 阶段：识别 `tl::AscendAllReduce<Reducer, threads, scale, thread_offset>::run`，按实例生成 PTO IR helper，然后把原调用点改写为 `func.call`。

PTO helper 的生成需要对齐 `reduce.h::run` 的分支语义：

- `threads <= scale`：直接返回输入；
- `threads <= 32` 且 `threads/scale` 适合寄存器规约：生成 warp 内 `pto.redux_*` 或 shuffle 规约；
- `threads > 32` 且满足 `pow2(threads)`、`pow2(scale)`、`scale <= 32`：生成跨 warp 两级规约，包含 UB scratch 写入、`pto.syncthreads`、第二级规约、结果写回和广播读取；
- 其它形态：生成 UB fallback，对应 `ub_reduce` 的三段同步语义。

以 RMSNorm 的 `SumOp, 128, 1, 0` 为例，helper 可以按 `cross_warp_reduce` 路径生成：

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

  %wid = arith.divui %laneid, %c32 : i32
  %lid = arith.remui %laneid, %c32 : i32

  %warp_sum = pto.redux_add %x : f32 -> f32

  %is_warp_leader = arith.cmpi ult, %lid, %c1 : i32
  scf.if %is_warp_leader {
    %idx = arith.index_castui %wid : i32 to index
    pto.store %warp_sum, %scratch[%idx] : !pto.ptr<f32, ub>, f32
  }

  pto.syncthreads

  %in_first_warp = arith.cmpi ult, %laneid, %c32 : i32
  %partial = scf.if %in_first_warp -> f32 {
    %has_value = arith.cmpi ult, %lid, %c4 : i32
    %v = scf.if %has_value -> f32 {
      %idx = arith.index_castui %lid : i32 to index
      %tmp = pto.load %scratch[%idx] : !pto.ptr<f32, ub> -> f32
      scf.yield %tmp : f32
    } else {
      %zero = arith.constant 0.0 : f32
      scf.yield %zero : f32
    }
    %sum = pto.redux_add %v : f32 -> f32
    scf.yield %sum : f32
  } else {
    %zero = arith.constant 0.0 : f32
    scf.yield %zero : f32
  }

  pto.syncthreads

  %is_writer = arith.cmpi ult, %laneid, %c1 : i32
  scf.if %is_writer {
    pto.store %partial, %scratch[%c0] : !pto.ptr<f32, ub>, f32
  }

  pto.syncthreads

  %result = pto.load %scratch[%c0] : !pto.ptr<f32, ub> -> f32
  return %result : f32
}
```

要点：

- helper 按 `<reducer, dtype, threads, scale, thread_offset>` 实例化，同一实例在 module 内只生成一次；
- `threads / scale / thread_offset` 都是编译期常量，PTO 后端直接从调用名解析实例参数；
- `SumOp / MaxOp / MinOp` 映射到对应的 `pto.redux_add / pto.redux_max / pto.redux_min` 或 UB fallback 标量组合；
- scratch pointer 直接来自最终 TIR 的调用参数，不由 PTO 后端重新推导固定 offset；
- `pto.syncthreads` 的插入点需要对齐 `reduce.h` 中 `asc_syncthreads()` 的可见性边界。

后续如果接入 PTODSL，可以在 PTODSL 中沉淀同样的 allreduce helper，再由 codegen 按实例选择 import / call。PTO 后端最终仍只消费普通 PTO IR。

### 3.5 TIR -> PTO 映射表

| 最终 TIR 构造 | PTO 后端目标表示 | 说明 |
| --- | --- | --- |
| `main_kernel` | `func.func @main_kernel ... attributes {pto.aicore}` | 外层 AICORE kernel |
| `SIMT_VF` block | `pto.simt_launch @body<<<dim_x, dim_y, dim_z>>>(...)` + `func.func @body ... attributes {pto.simt_entry}` | 使用结构化 SIMT launch |
| `simtvf_tx` / `threadIdx.x` | `pto.get_tid_x` | SIMT body 内获取 workitem 坐标 |
| 标量常量、算术、比较 | `arith.*` | 基础标量 IR |
| `for` / `if` | `scf.for` / `scf.if` | 基础结构化控制流 |
| 连续 `float2` 访存循环 | LLVM Dialect 向量 load/store | 保留连续 2-lane bundle 信息 |
| `x_frag[32]` | lane-local scratch，允许后续 SROA 消除 | 对齐 Ascend 路径的 workitem-local fragment |
| `tl::AscendAllReduce<...>::run` | 自动实例化的特化 PTO IR helper + `func.call` | helper 内部使用 `pto.redux_add + pto.syncthreads + pto.load/store` |
| `rsqrt` | `sqrt + reciprocal/div` | 无需先扩展 PTO 标量 op 面 |
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

### 4.2 连续向量访存 lowering

需要做的事：

1. 识别 `BufferLoad/BufferStore` 上 `Ramp(base, 1, lanes)` 这种连续 lane bundle；
2. `d=4096` 的 `float2` 访存映射到 LLVM Dialect `vector<2xf32>` load/store；
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
2. **LLVM 级**：检查 `llvm.hivm.sync.workitems`、`<2 x float>` load/store，以及后续的 `extractelement`；
3. **功能与性能级**：先以 RMSNorm 作为第一条 vertical slice，再扩展到其它 `SimtVF` kernel。

---

## 结论

对 TileLang RMSNorm kernel 而言，对接 PTO 后端的要点是：以最终 device TIR 为输入；`SIMT_VF` 映射到 `pto.simt_launch + pto.simt_entry`；连续 `float2` 访存保留为 MLIR/LLVM `vector<2xf32>` load/store；`tl::AscendAllReduce<...>::run` 用自动实例化的特化 PTO IR helper 承接。

`feature-simt-ops` 上已经具备 `pto.syncthreads -> llvm.hivm.sync.workitems`，因此两级 allreduce 的 PTO IR 表达所需的关键同步原语已经齐了。

整体方向是：TileLang 侧把高层语义展开清楚；PTO 后端编译普通 PTO IR 加必要的 LLVM 向量访存；向量 fragment 继续复用 Ascend 路径验证过的 `vector<2xf32>` + SROA 优化链。
