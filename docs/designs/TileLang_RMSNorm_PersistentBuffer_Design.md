# TileLang RMSNorm:基于 persistent buffer 标记的寄存器驻留优化

## 概述

要解决的问题和 [`TileLang_RMSNorm_PersistentFragment_Design.md`](./TileLang_RMSNorm_PersistentFragment_Design.md) 一样:RMSNorm 里权重 `W` 对一个核内部的 token 循环是不变量,`w_ub` 已经常驻 UB,但每个 token 进入 SimtVF 算输出时还要把 `w_ub` 从 UB 读进寄存器一次,每核重复 `batch/N_CORES` 次。

这一版换一个角度。前端不写 fragment、不写 init-VF,只在 `alloc_shared` 上加一个 `persistent=True`,声明这个 buffer 初始化后内容不再改;编译器据此识别出对它的重复读,自动把读的结果固化到寄存器。底层仍然用 PTOAS 的 `pto.keep` / `pto.resume`,寄存器分配仍归 BiSheng,我们只生成 keep/resume。

它和上一版(显式 fragment)是两个并列的备选,底层机制相同,差别在前端表达和编译器要做的工作,详见 §7 对比。

## 1. 问题

完整的 example、生成 ASC、tiling 策略、每 token 计算方式,见基础稿 [`TileLang_RMSNorm_to_PTO_Design.md`](./TileLang_RMSNorm_to_PTO_Design.md) 和上一版 §1,这里只点一下要消掉的那段开销:输出行 `y[i] = x_frag[i] * rstd * w_ub[i]` 每 token 都把整段 `w`(每线程 32 个 f32)从 UB 读进寄存器,而 `w` 在整个 token 循环里不变。目标是把这段 `UB→VRF` 从每核 64 次降到 1 次。

`keep` / `resume` 的语义(slot 映射到固定物理寄存器 `R4..R126`,用 inline-asm `MOV` 承载,硬件在 VF 调用之间保留寄存器状态),见上一版 §4.1 与 PTOAS 的 `vpto-simt-keep-resume-design.md`,本文不再展开。

## 2. 前端:`persistent=True`

```python
def main(X, Y, W, RSTD, eps):
    n_tokens_per_core = batch // N_CORES
    with T.Kernel(N_CORES) as core_id:
        w_ub      = T.alloc_shared((d,), dtype, persistent=True)   # ← 只加这一个 kwarg
        x_ub      = T.alloc_shared((TILE,), "float32")
        y_ub      = T.alloc_shared((TILE,), "float32")
        z_rstd_ub = T.alloc_shared((8,), "float32")

        T.copy(W[:d], w_ub[:d])                 # GM→UB:W 只搬一次(这是它的 init)

        for t in T.Pipelined(n_tokens_per_core, num_stages=2):
            base = (t * N_CORES + core_id) * d
            T.copy(X[base : base + d], x_ub[:d])
            with T.SimtVF(threads=threads):
                x_frag = T.alloc_fragment((TILE,), "float32")
                for i in T.Parallel(TILE):
                    x_frag[i] = x_ub[i]
                sum_sq = T.alloc_reducer((1,), "float32", op="sum", replication="all")
                T.clear(sum_sq)
                for i in T.Parallel(TILE):
                    if i < d:
                        sum_sq[0] += x_frag[i] * x_frag[i]
                T.finalize_reducer(sum_sq)
                rstd_val = T.rsqrt(sum_sq[0] / d + eps)
                z_rstd_ub[0] = rstd_val
                for i in T.Parallel(TILE):
                    if i < d:
                        y_ub[i] = x_frag[i] * rstd_val * w_ub[i]   # ← 仍然直接读 w_ub,body 不动
            row_id = t * N_CORES + core_id
            T.copy(y_ub[:d], Y[base : base + d])
            T.copy(z_rstd_ub[:1], RSTD[row_id : row_id + 1])
    return main
```

`persistent=True` 的语义是:这个 buffer 在生命周期内被初始化一次,之后内容不再被修改,因此读它的结果可以固化在寄存器里、跨多次 SimtVF 复用。相比上一版,kernel body 一行不用改:不引入 `w_frag`,不写显式的 init-SimtVF,只多一个 kwarg。

这个标记是用户对数据的承诺,用错了会出隐蔽的数值错误(寄存器里缓存的是旧值),所以编译器要做一个轻量验证:init 那次 `T.copy` 之后,kernel 内不应再出现对这个 buffer 的写(store / copy)。验证失败就报错,或者忽略 `persistent`、退回普通 buffer 每次从 UB 现读。

## 3. TIR 形态

`persistent` 作为属性记录下来、一路保留到 codegen。和上一版最大的不同是 TIR 不重构:不 hoist fragment、不加 init-SimtVF、不改 `layout_map`,`SIMT_VF` block 原样,输出循环里仍是对 `w_ub` 的读。相对现状的最终 device TIR(见基础稿 §2.1),只多一个 persistent 标记:

```python
@T.prim_func
def main_kernel(RSTD, W, X, Y, eps):
    w_ub = T.decl_buffer((4096,), data=buf_dyn_shmem, scope="shared.dyn")   # 仍是普通 shared buffer
    T.attr(w_ub, "tl.persistent", 1)                             # ← persistent 标记,挂在 buffer 上
    ...
    bx = T.launch_thread("blockIdx.x", 64)
    T.ascend_copy_gm_to_ubuf(w_ub, W, ...)                       # init:W 的 GM→UB,只一次

    for t in range(64):
        T.ascend_copy_gm_to_ubuf(x_ub, X[...], ...)             # MTE2
        with T.block("SIMT_VF", no_realize=True):               # body 原样不动
            T.block_attr({"layout_map": {x_frag: metadata["tl.Fragment"][0], ...}})
            x_frag = T.allocate([32], "float32", "local")
            for i in T.unroll(8):
                x_frag[i*4:i*4+4] = x_ub[...]
            # ... Σx² + tl::AscendAllReduce + rstd ...
            for i in T.unroll(8):
                y_ub[...] = x_frag[i*4:i*4+4] * T.Broadcast(rstd_val, 4) \
                          * w_ub[i*512 + simtvf_tx*4 : ...]       # ← 仍读 w_ub(已标 persistent)
        T.ascend_copy_ubuf_to_gm(Y[...], y_ub, ...)             # MTE3
```

persistent 作为 buffer 属性挂在 `w_ub` 上(`tl.persistent`),和前端 `alloc_shared(persistent=True)` 对应。codegen 据它在 PTO 侧给对应 UB 指针带上 persistent 属性(§4.1 的 `!pto.ptr<f32, ub> {persistent}`),之后的 peel 和 keep/resume 都在 PTO 做(§4),TIR 这层不动结构。

## 4. PTO 对接:simtscope → peel → keep/resume → outline

核心思路:TIR 接到 PTO 时,先不把 `SIMT_VF` block 直接变成一个 outline 出去的 `simt_entry` 函数,而是表示成一个内联的 region `pto.simtscope`。peel 和插 keep/resume 这些控制流变换都在这个内联形态上做,做完再把 `pto.simtscope` outline 成 `simt_entry` 函数。这样变换阶段一切都在 `pto.aicore` 这一个函数里、控制流可见,不用做跨函数变换。下面的 MLIR 为讲解把 `w_ub` 当成独立的 persistent 指针;实际 device TIR 里它是 `buf_dyn_shmem` 的一个 view,这个差异和影响见 §4.5。

### 4.1 第一步:TIR→PTO,SIMT_VF 转成内联 `pto.simtscope`

```mlir
func.func @rms_kernel(...) attributes {pto.aicore} {
  %w_ub = ... : !pto.ptr<f32, ub>  {persistent}     // persistent 属性挂在 UB 指针上
  // ascend_copy_gm_to_ubuf(%w_ub, W) ...           // init:W 的 GM→UB,只一次
  scf.for %t = 0 to 64 step 1 {
    // mte_gm_ub %x_ub ...
    pto.simtscope {                                  // ← SIMT_VF block:内联 region,尚未 outline
      %tx = pto.get_tid_x : i32
      // load x_frag、Σx² + all_reduce、rstd ...
      scf.for %i = 0 to 8 step 1 {
        %w = pto.load %w_ub[%i*512 + %tx*4] : vector<4xf32>     // 读 persistent UB 指针
        // %y = %xf * %rstd * %w ; pto.store %y, %y_ub[...]
      }
    }
    // mte_ub_gm %y_ub ...
  }
  return
}
```

`pto.simtscope` 是一个带 region 的 op,语义等价于 `SIMT_VF` block(SIMT 线程并行执行其 body),但它还在外层函数体内,不是一个独立函数。它承载了原来要 outline 的内容,只是把 outline 推迟到所有变换之后。

### 4.2 第二步:识别

在 `pto.simtscope` 内找对带 `persistent` 属性的 UB 指针的读,满足两个条件就触发:

- 这个 simtscope 被一个 `scf.for` 反复执行(对应循环里反复调用同一个 SimtVF);
- 读地址与循环变量无关,即每次迭代读的是同一组元素。RMSNorm 里 `%w_ub[%i*512 + %tx*4]` 只依赖 `tx` 和展开后的 `i`,不依赖 `t`,是 loop-invariant 的。

`persistent` 是这个识别成立的前提:没有它,编译器不能假设 `w_ub` 的内容在迭代之间不变(中间可能被别处写),也就不能把读提出来。

连续多个不同 simtscope 读同一 persistent 指针的情形(非循环)见 §6。

### 4.3 第三步:peel + keep/resume(在内联形态上做)

peel `scf.for` 的首次迭代,得到一个 peeled simtscope(iter0)和剩余迭代的 simtscope。peeled iter0 里 `w` 仍从 UB 读,scope 末尾 `keep`;剩余迭代的 simtscope 入口把对 `w` 的读换成 `resume`,scope 末尾 `re-keep`。内层 `scf.for %i` 照常 unroll,M 条 keep/resume 各压成一条 `keep_range` / `resume_range`(同上一版)。

```mlir
func.func @rms_kernel(...) attributes {pto.aicore} {
  // ascend_copy_gm_to_ubuf(%w_ub, W) ...

  // peeled iter0:读 UB + 计算 + keep
  // mte_gm_ub %x_ub (token 0) ...
  pto.simtscope {
    %tx = pto.get_tid_x : i32
    %w0..%w31 = <8 个 vector<4xf32> load from %w_ub>     // 仍从 UB 读 w
    // ... load x_frag、rstd、用 %wk 算 y[0] ...
    pto.keep_range %w0, ..., %w31 {slot_base = 0, n = 32}    // 末尾 keep
  }
  // mte_ub_gm %y_ub (token 0) ...

  scf.for %t = 1 to 64 step 1 {                          // 剩余迭代
    // mte_gm_ub %x_ub (token t) ...
    pto.simtscope {
      %w0..%w31 = pto.resume_range {slot_base = 0, n = 32}   // 入口 resume,不再读 UB
      // ... load x_frag、rstd、用 %wk 算 y[t] ...
      pto.keep_range %w0, ..., %w31 {slot_base = 0, n = 32}  // 末尾 re-keep
    }
    // mte_ub_gm %y_ub (token t) ...
  }
  return
}
```

每个 consumer(含末次迭代)scope 末尾都 re-keep,不对最后一次做特判。原因和实现取舍见上一版 §4.3、§4.5:resume 之后 body 可能覆盖那几个 slot 寄存器,只要之后还有迭代会 resume 同一 slot,这次就得在末尾把它复位;末次那条是死代码,但省掉它要给循环体做特判、反而更复杂,而且 read-only 下它是自移动、代价可忽略。

peel 之后,`w_ub` 只在 peeled iter0 里被读一次,剩余循环体不再读它。

### 4.4 第四步:outline `pto.simtscope` → `simt_entry`

变换做完,把每个 `pto.simtscope` region outline 成一个 `func.func {pto.simt_entry}`,在原位置换成 `pto.simt_launch`:

```mlir
func.func @rms_kernel(...) attributes {pto.aicore} {
  // ascend_copy_gm_to_ubuf(%w_ub, W) ...
  pto.simt_launch @rms_iter0<<<128,1,1>>>(%w_ub, %x_ub, %y_ub, ...)   // peeled iter0
  scf.for %t = 1 to 64 step 1 {
    pto.simt_launch @rms_body<<<128,1,1>>>(%x_ub, %y_ub, ..., %t)     // 剩余迭代(不传 w_ub)
  }
  return
}

func.func @rms_iter0(%w_ub: !pto.ptr<f32, ub>, ...) attributes {pto.simt_entry} {
  %tx = pto.get_tid_x : i32
  %w0..%w31 = <8 个 float4 load from %w_ub>
  // ... 算 y[0] ...
  pto.keep_range %w0, ..., %w31 {slot_base = 0, n = 32}
  return
}

func.func @rms_body(..., %t: i32) attributes {pto.simt_entry} {
  %w0, ..., %w31 = pto.resume_range {slot_base = 0, n = 32}
  // ... 算 y[t] ...
  pto.keep_range %w0, ..., %w31 {slot_base = 0, n = 32}     // re-keep
  return
}
```

`keep_range` / `resume_range` 最终展开成定址 `R4..R126` 的 `MOV`(slot 编号由这个 pass 从 `R4` 起分配)。`w_ub` 仍是普通 UB buffer,只在 `@rms_iter0` 里读一次;循环体 `@rms_body` 不再有 `w` 的 `UB→VRF`。

这一版在 PTO IR 上要加的东西比上一版少:不需要 `lane_persistent` 地址空间、`pto.persistent_lane_alloc` 那个承载体、以及对应的 cleanup;`persistent` 只是 UB 指针上的一个识别用属性,keep/resume 直接落在 slot 上。代价是多了 `pto.simtscope` 这个 op 和晚 outline 的 pass,以及识别 + peel 的分析。

### 4.5 实际生成的 TIR 与方案预期的差异

§4.1 到 §4.4 的 MLIR 把 `w_ub` 当成一个干净的 `!pto.ptr<f32, ub> {persistent}` 指针,是为了讲清流程。实跑的 device TIR(`48_final_device_mod.tir`,PTO 后端实际消费的那层)不是这个形态:`MergeSharedMemoryAllocations` 已经把各 shared buffer 合并进一块 `buf_dyn_shmem`,`w_ub` 只是它上面的一个 view。下面是省略无关部分后的真实框架:

```python
@T.prim_func
def main_kernel(RSTD, W, X, Y, eps):
    T.func_attr({..., "dyn_shared_memory_buf": 82496, ...})
    buf_dyn_shmem = T.handle("uint8", "shared.dyn")
    # 所有 shared buffer 都是同一块 buf_dyn_shmem 的 view:
    w_ub      = T.decl_buffer((4096,), data=buf_dyn_shmem, scope="shared.dyn")   # 偏移 0
    y_ub      = T.decl_buffer((8192,), data=buf_dyn_shmem, scope="shared.dyn")
    x_ub      = T.decl_buffer((8192,), data=buf_dyn_shmem, scope="shared.dyn")
    z_rstd_ub = T.decl_buffer((16,),   data=buf_dyn_shmem, scope="shared.dyn")
    bx = T.launch_thread("blockIdx.x", 64)
    buf_dyn_shmem = T.allocate([82496], "uint8", "shared.dyn")                   # 合并后的整块 arena
    tx = T.launch_thread("threadIdx.x", 128)

    # init:把 W 搬进 buf_dyn_shmem 的 [0,4096)。注意目标写的是 buf_dyn_shmem+0,不带 w_ub 名字
    T.ascend_copy_gm_to_ubuf(T.tvm_access_ptr("float32", buf_dyn_shmem, 0, 4096, 2),
                             T.tvm_access_ptr("float32", W, 0, 4096, 1), 0, 1, 512, 0, 0)
    T.ascend_set_flag("V_MTE2", 0)                                              # flag 初始化(略)

    for t in range(64):
        T.ascend_wait_flag("V_MTE2", t % 2)
        T.ascend_copy_gm_to_ubuf(T.tvm_access_ptr("float32", buf_dyn_shmem, t%2*4096+4096, 4096, 2),
                                 T.tvm_access_ptr("float32", X, t*262144+bx*4096, 4096, 1), ...)   # MTE2:载 x
        T.ascend_set_flag("MTE2_V", t % 2); T.ascend_wait_flag("MTE2_V", t % 2)
        T.ascend_wait_flag("MTE3_V", t % 2)
        with T.block("SIMT_VF", no_realize=True):
            T.block_attr({"layout_map": {x_frag_2: metadata["tl.Fragment"][0],
                                         sum_sq_2: metadata["tl.Fragment"][1]}})
            simtvf_tx = T.launch_thread("threadIdx.x", 128)
            T.attr("simtvf", "tl.simtvf_scope", 1)
            x_frag   = T.allocate([32], "float32", "local")
            x_frag_3 = T.Buffer((32,), data=x_frag, scope="local")
            for i in T.unroll(8):                                              # 载 x → x_frag
                x_ub_1 = T.Buffer((8192,), data=buf_dyn_shmem, scope="shared.dyn")
                x_frag_3[i*4:i*4+4] = x_ub_1[t%2*4096 + i*512 + simtvf_tx*4 + 4096 : ...]
            # ... Σx² + tl::AscendAllReduce + rstd_val(略)...
            for i in T.unroll(8):                                              # 输出,这里读 w
                y_ub_1 = T.Buffer((8192,), data=buf_dyn_shmem, scope="shared.dyn")
                w_ub_1 = T.Buffer((4096,), data=buf_dyn_shmem, scope="shared.dyn")   # ← block 内重新 decl 的 w view
                y_ub_1[t%2*4096 + i*512 + simtvf_tx*4 + 12288 : ...] = \
                    x_frag_3[i*4:i*4+4] * T.Broadcast(rstd_val, 4) \
                    * w_ub_1[i*512 + simtvf_tx*4 : ...]                         # ← 读 w_ub_1(仍是 buf_dyn_shmem 的 view)
        T.ascend_set_flag("V_MTE3", t % 2)                                     # MTE3:写回 y / rstd(略)
```

**关键差异(预期 IR vs 实际 TIR):**

| | §4.1–§4.4 的预期 IR | 实际 device TIR |
|---|---|---|
| w 的载体 | 独立 `!pto.ptr<f32, ub> {persistent}` | `buf_dyn_shmem` 的一个 view(偏移 0) |
| W 的 init 写入 | 写 `%w_ub` | `tvm_access_ptr(buf_dyn_shmem, 0, ...)`,无 w_ub 名字 |
| SimtVF 内读 w | `pto.load %w_ub[...]` | `w_ub_1[...]`(block 内重新 decl 的 view) |
| persistent 归属 | 一个 buffer | `buf_dyn_shmem` 的 `[0, d)` region |

**缺少的信息 / 落地必须补的:**

1. **region 身份**:IR 里没有"persistent buffer"这个一等对象,只有 `buf_dyn_shmem + offset`。要把 persistent 记成"`buf_dyn_shmem` 的 `[base, base+d)` 这段 region",识别时按"读是否落在这段 region 内"来判,而不是按 buffer 名字。
2. **region 级 immutability 验证**:要按地址区间 / 别名查写——别的 view 写到 `[base, base+d)`(如 `buf_dyn_shmem[0:d] = ...` 或另一个 alias)同样破坏不变性,而名字里看不到 `w_ub`。
3. **属性透传扛过 merge**:前端 `alloc_shared(persistent=True)` 的标记要 propagate 到 merge 之后那个内联 re-decl 的 `w_ub_1` view 上(它是新的 Buffer 对象),否则 PTO 这层就丢了识别依据。
4. **宜在 merge 之前 capture**:merge / flatten 之后只剩 `base + offset`,逻辑身份难恢复(基础稿 §10.2 是同一类问题)。所以 persistent 的识别和 immutability 验证最好在 merge 之前做(那时 `w_ub` 还是独立 buffer、读写无歧义),把结论(哪段 region persistent、哪些读要 hoist)作为 metadata 带过 merge;peel + keep/resume 仍在 PTO 做,靠这份 metadata,不必在裸 offset 上重新猜。

一句话:§4.1–§4.4 的 `%w_ub {persistent}` 是讲解用的理想形态,落地时它对应的是"`buf_dyn_shmem` 的某段 region 带 persistent",验证、属性透传、识别都要按 region 来。

## 5. 寄存器预算与 fallback

和上一版一样:缓存整段 `w` 要占 `M = TILE/threads` 个 slot(`R4..R126` 共 123 个)。判据是 `w 的 slot 数 + body 峰值寄存器需求 + 余量 ≤ 123`;放不下时不做 peel,把对 `w_ub` 的读原样留着,退回每 token 现读。RMSNorm 的 body 本就占掉半个寄存器堆,是这个优化的边缘 case,RoPE 的 cos/sin、LayerNorm/GroupNorm 的 bias 是更合适的首发载体。细节见上一版 §4.7。

## 6. 连续多个 SimtVF 的情形(简述)

如果对同一个 persistent UB 指针的重复读来自几个连续的、不同的 simtscope(而非一个循环反复调用),处理方式类似但不用 peel:把对这个指针的读抽出来,在它们前面合成一个单独的 simtscope,只做读 + `keep`;后面那些 simtscope 把对它的读换成 `resume`,每个非末尾的在末尾 `re-keep`。这就是 keep/resume 的接力(见上一版 §4.5),循环只是其中一种特例。

## 7. 与上一版(显式 fragment)的对比

两版要解决的问题、底层机制相同,差别如下。

| | 上一版(显式 fragment) | 本版(persistent buffer) |
|---|---|---|
| 前端 | 在 SimtVF 外 `alloc_fragment` + 手写 init-SimtVF,输出改读 `w_frag` | `alloc_shared(..., persistent=True)`,body 一行不改 |
| TIR | 重构:hoist `w_frag`、加 init-SimtVF、共享 `layout_map` | 不重构:body 原样,只多一个 persistent 标记 |
| 谁做 hoist | 用户显式写出来 | 编译器识别 loop-invariant 的 persistent 读,自动 peel |
| PTO IR 增量 | `lane_persistent` 地址空间、`pto.persistent_lane_alloc`、cleanup pass | `pto.simtscope`(晚 outline)、UB 指针上的 `persistent` 属性 |
| 底层机制 | `keep`/`resume`,slot → `R4..R126` | 同 |
| re-keep | 每个非末尾 consumer 出口 re-keep | 同(且本版不特判末次) |
| 产出结构 | init-VF + 统一的循环体(64 次一致) | peeled iter0 + 剩余循环体(body 复制一份) |
| 主要代价 | 用户要改 kernel 结构;编译器较简单 | 前端只加 kwarg;编译器要做识别 + peel + 晚 outline |

一句话:上一版把活儿放在前端(用户手动 hoist),本版把活儿放在编译器(persistent 标记驱动自动 peel)。两版的 keep/resume 落地完全一样,可以共用那部分实现。

## 8. 落地步骤与开放问题

落地顺序:

1. 前端 `alloc_shared` 支持 `persistent` kwarg;加 init 之后无写的轻量验证 + 失败退化。
2. TIR 透传 persistent 属性到 codegen。
3. PTO:新增 `pto.simtscope` op 和把它 outline 成 `simt_entry` 的 pass;先让普通(无 persistent)路径跑通,即 simtscope 直接 outline,行为和现状一致。
4. PTO:识别 loop-invariant 的 persistent UB 读 + `scf.for` 首次迭代 peel + 插 keep/resume(`keep_range`/`resume_range`)。
5. 寄存器预算检查 + fallback。
6. 首发用 RoPE(或 LayerNorm bias)验证,RMSNorm 因压力大、HBM-bound 主要用来对数值。

开放问题:

1. 寄存器压力:同上一版,`M + body 峰值 + 余量 ≤ 123` 的阈值和 fallback 粒度要定。
2. peel 的 body 复制:peeled iter0 复制了一份完整计算 body,代码体积变大;若在意,可考虑 §6 那种"抽取读成单独 simtscope"的做法替代 peel(产出和上一版趋同),作为后续优化。
3. 识别精度:loop-invariant 判定要稳妥,地址依赖循环变量时不能 hoist;persistent 验证漏判会出隐蔽数值错。
4. `pto.simtscope` 与现有 PTO pass 的交互:晚 outline 改变了 SimtVF 的出现时机,要确认 flag/DMA、pipeline 等 pass 不受影响。
