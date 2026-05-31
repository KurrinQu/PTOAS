# TileLang RMSNorm:跨 SimtVF 寄存器驻留的优化方案

## 概述

RMSNorm kernel 里,权重 `W` 对一个核内部的 token 循环是不变量。当前实现已经把 `W` 的 `GM→UB` 搬运提到循环外(`w_ub` 常驻 UB),但每个 token 进入 SimtVF 算输出时,还要把 `w_ub` 从 UB 读进寄存器一次,每核重复 `batch/N_CORES` 次。

本方案让 fragment 跨 SimtVF 调用驻留在寄存器里:`w` 一次性读进寄存器,之后所有 token 复用,这段 `UB→VRF` 从每核 64 次降到 1 次。用的是 PTOAS 已有的 `pto.keep` / `pto.resume`,它把一个值钉在固定物理寄存器 `R4..R126` 上,用 inline-asm `MOV` 承载。寄存器分配仍归 BiSheng,我们只生成 keep/resume。

收益的边界要说清楚:这个 kernel 是 HBM 带宽受限的,省掉这次 UB 读大概率不改变端到端时间;可能见效的是缓解 UB 端口争用,还有功耗。它更大的价值是作为一个通用能力,服务 RoPE 的 cos/sin、LayerNorm/GroupNorm 的 weight/bias 这类整段循环不变的小张量。

全文四部分:现状、优化机会、前端/TIR 方案、PTO IR 对接。

---

## 1. 现状:当前 RMSNorm example 与生成代码

### 1.1 Example 实现

以 `examples/ascend/example_rmsnorm_auto.py` 为准(issue #29 引用的版本)。核心结构:

```python
def rms_norm_fwd(batch, d, dtype="float32"):
    N_CORES = 64
    TILE = tilelang.next_power_of_2(d)
    threads = 256 if TILE > 4096 else 128          # 自适应:~32 元素/线程
    N = batch * d

    @T.prim_func
    def main(X, Y, W, RSTD, eps):
        n_tokens_per_core = batch // N_CORES        # 4096/64 = 64
        with T.Kernel(N_CORES) as core_id:
            w_ub      = T.alloc_shared((d,), dtype)
            x_ub      = T.alloc_shared((TILE,), "float32")
            y_ub      = T.alloc_shared((TILE,), "float32")
            z_rstd_ub = T.alloc_shared((8,), "float32")

            T.copy(W[:d], w_ub[:d])                 # GM→UB:W 只搬一次

            for t in T.Pipelined(n_tokens_per_core, num_stages=2):
                base = (t * N_CORES + core_id) * d
                T.copy(X[base : base + d], x_ub[:d])

                with T.SimtVF(threads=threads):
                    x_frag = T.alloc_fragment((TILE,), "float32")
                    for i in T.Parallel(TILE):
                        x_frag[i] = x_ub[i]                       # UB→VRF:x(每 token 必须)

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
                            y_ub[i] = x_frag[i] * rstd_val * w_ub[i]   # ← 每 token 从 UB 读 w
                row_id = t * N_CORES + core_id
                T.copy(y_ub[:d], Y[base : base + d])
                T.copy(z_rstd_ub[:1], RSTD[row_id : row_id + 1])
    return main
```

### 1.2 Tiling 策略

以 `d=4096, batch=4096`(`TILE=4096, threads=128`)为例:

- **Grid**:`T.Kernel(N_CORES=64)` 启动 64 个 AIV 向量核,每核串行处理 `batch/64 = 64` 个 token。
- **Token→core 交错映射**:`row_id = t*64 + core_id`,`base = row_id*d`。第 t 步时 64 个核一起处理连续的 `[t*64, t*64+64)` 这 64 行,跨核读到 GM 上一整块 `64*d` 连续地址,利于 HBM 合并访问。生成码即 `X[t*262144 + bx*4096]`(`262144=64*d`,`4096=d`)。
- **隐藏维不切块**:整行 `d` 一次进 UB tile,`TILE=next_pow2(d)` 便于 float4 向量化与线程整除;非 2 次幂时 `[d,TILE)` 为 padding,由 `if i < d` 掩掉。
- **双缓冲 + 三级流水**:`T.Pipelined(num_stages=2)` 自动把 `x_ub`/`y_ub` 双缓冲,稳态下 `MTE2(GM→UB)`、`VEC(计算)`、`MTE3(UB→GM)` 三引擎并行,靠 `set_flag`/`wait_flag` 握手。
- **线程自适应**:`TILE/threads` 恒为 32(`4096/128` 或 `8192/256`),每线程 8 个 float4。

### 1.3 每 token 的计算方式(SimtVF 内)

128 个 SIMT 线程协作处理一行 4096 元素,核心是 fragment trick:x 一次性进寄存器,reduce 与输出两次复用。

1. **float4 向量化 load**:`x_frag[32]`,每线程 8 个 float4;线程 tx 取 `i*512 + tx*4`(`512=128×4`)处连续 4 元素,8 组覆盖 4096。
2. **行内 `Σx²`**:每线程先对自己 32 个数求平方和(从寄存器,不再访 UB),再 `AscendAllReduce` 跨 128 线程规约并广播。
3. **rstd**:`rstd = rsqrt(Σx²/d + eps)`,写入 `z_rstd_ub`。
4. **输出**:`y[i] = x_frag[i] * rstd * w_ub[i]`,float4 写回 `y_ub`。这里复用了寄存器里的 x,但 w 是从 UB 现读的。

### 1.4 生成的 ASC 代码(关键片段)

`T.SimtVF` body 被渲染成 helper `simt_vf_0`,用到的 buffer 经 `tl.simt_vf_captures` 当入参捕获传入(`buf_dyn_shmem`);`alloc_fragment` 在 helper 内是每次调用重建的本地寄存器数组 `float x_frag[32]`:

```cpp
__simt_vf__ __launch_bounds__(128) inline void simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t t, float eps) {
  float x_frag[32];
  float sum_sq[1];
  // ① float4 load x → 寄存器
  for (int i = 0; i < 8; ++i)
    *(float4*)(x_frag + i*4) = *(__ubuf__ float4*)(((__ubuf__ float*)buf_dyn_shmem) + ((t&1)*4096 + i*512 + simtvf_tx*4 + 4096));
  // ② 局部平方和 + 跨线程规约
  sum_sq[0] = 0.f;
  for (int i = 0; i < 32; ++i) sum_sq[0] += x_frag[i]*x_frag[i];
  sum_sq[0] = tl::AscendAllReduce<tl::SumOp, 128, 1, 0>::run(sum_sq[0], &((__ubuf__ float*)buf_dyn_shmem)[20480]);
  // ③ rstd
  float rstd_val = rsqrtf(sum_sq[0]/4096.f + eps);
  ((__ubuf__ float*)buf_dyn_shmem)[((t&1)*8 + 20608)] = rstd_val;
  // ④ 输出 y = x * rstd * w  ← 末项每 token 从 UB 现读 w_ub
  for (int i = 0; i < 8; ++i)
    *(__ubuf__ float4*)(...y_ub...) =
        (*(float4*)(x_frag + i*4) * make_float4(rstd_val,rstd_val,rstd_val,rstd_val))
        * *(__ubuf__ float4*)(((__ubuf__ float*)buf_dyn_shmem) + (i*512 + simtvf_tx*4));   // ← w_ub 现读
}
```

外层 `main_kernel` 在 `for (t...)` 里每 token 调一次 `asc_vf_call<simt_vf_0>(...)`。所以输出那一步的 `w_ub` 现读,每核要发生 `n_tokens_per_core = 64` 次。

---

## 2. 优化机会与优化想法

### 2.1 机会

`w_ub` 已在 UB 常驻,但输出那一步每 token 都把整段 `w`(每线程 32 个 f32)重新从 UB 读进寄存器。语义上想表达的是“W 整段循环内不变,搬进寄存器后一直留着”,而 DSL/IR 目前没有这个能力,只能保守地每次进 VF 都重读。

| 项目 | 当前 | 目标 |
|---|---|---|
| `GM→UB`(W) | kernel 内 1 次 | 不变 |
| `UB→VRF`(w) | 每 token 1 次,共 `batch/N_CORES` 次 | kernel 内 1 次 |
| VRF 占用 | 仅 SimtVF 内瞬时 | kernel 期间常驻 `d*sizeof(dtype)` |

### 2.2 硬件 / 工具链基础

Ascend 950 的 SIMT 寄存器状态在两次连续 SimtVF 调用之间不会被自动清零:前一个 VF 写入的逻辑值,可在后一个 VF 中稳定读取。PTOAS 已把这层语义建模为 `pto.keep` / `pto.resume`,这是一对靠编译期常量 `slot` 配对的 op,`slot` 直接映射到固定物理寄存器 `R4..R126`,lowering 成 inline-asm `MOV` 的 sideeffect call。

寄存器分配由 BiSheng(LLVM 层)做,在 PTO 之下,我们改不了。`keep`/`resume` 用的是固定 `R` 操作数加 sideeffect 的 inline asm,BiSheng 在这个点会绕开,VF 之间的寄存器保留交给硬件。所以本方案只生成 keep/resume,不写任何寄存器分配或预留。

这个特性 PTOAS 已经做好,不需要新硬件或 runtime。动手前最好先用一个最小用例验证 keep/resume 的跨调用语义:相邻两个 simt_entry,一个存、一个取,值不变。

### 2.3 收益评估

这个 kernel 已经打到 HBM 峰值的 80–88%,是带宽受限的。三级流水里 VEC(含 UB 读)被 HBM 搬运盖住:`d=4096` 时 HBM 侧约 1.5 us/token,VEC 侧约 0.1–0.2 us,VEC 有近 10 倍余量。所以单看 wall-clock,砍掉 w 的这次 UB 读大概率没区别。

更可能见效的是 UB 读写端口争用。稳态下 MTE2 写 x、MTE3 读 y、VEC 同时读 x、读 w、写 y,都压在 UB 上;如果瓶颈其实是 UB 端口而非 HBM(那 12% 的 gap 里有一部分可能来自这里),释放 VEC 的 w 读才有意义。这点值得先 profiling 验证一下。

通用价值更明确:RoPE 的 cos/sin(head_dim 内所有 token 共享)、LayerNorm/GroupNorm 的 weight/bias、MoE expert-level 的常量偏置。RoPE 比“RMSNorm 的 W”更能体现价值(每 token 多次复用、表更小),适合作为主推例子。即便不省时间,少 63/64 的 w UB 读也省能耗。

所以它的定位是通用原语;别只盯着 RMSNorm 这张 benchmark 的数字,先量后改。

---

## 3. 前端 / TIR 优化方案与形态

### 3.1 前端写法:alloc 在 SimtVF 外即视为 persistent

采纳 issue #29 评论区的共识:不新增任何前端 API,连 `persistent=True` kwarg 也不要。fragment 只要 alloc 在 SimtVF 外层(core 作用域),编译器就据 alloc 位置推断它需要跨 VF 寄存器驻留;初始化复用现有的 `T.copy`。下面是把 init 显式包成一个 SimtVF 的去糖形式,lowering 最直白:

```python
with T.Kernel(N_CORES) as core_id:
    w_ub = T.alloc_shared((d,), dtype)
    x_ub = T.alloc_shared((TILE,), "float32")
    y_ub = T.alloc_shared((TILE,), "float32")
    z_rstd_ub = T.alloc_shared((8,), "float32")

    T.copy(W[:d], w_ub[:d])                              # (A) GM→UB,不变

    w_frag = T.alloc_fragment((TILE,), dtype)           # (B) alloc 在 SimtVF/循环外 ⇒ persistent
    with T.SimtVF(threads=threads):                     # (C) 一次性 UB→VRF 初始化(循环外只跑一次)
        for i in T.Parallel(TILE):
            if i < d:                                   #     防止读 w_ub 越界(w_ub 只有 d)
                w_frag[i] = w_ub[i]

    for t in T.Pipelined(n_tokens_per_core, num_stages=2):
        base = (t * N_CORES + core_id) * d
        T.copy(X[base : base + d], x_ub[:d])
        with T.SimtVF(threads=threads):
            x_frag = T.alloc_fragment((TILE,), "float32")   # 仍是被流水轮转的瞬时 fragment
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
                    y_ub[i] = x_frag[i] * rstd_val * w_frag[i]   # ← 用 w_frag,不再每 token UB→VRF
        row_id = t * N_CORES + core_id
        T.copy(y_ub[:d], Y[base : base + d])
        T.copy(z_rstd_ub[:1], RSTD[row_id : row_id + 1])
```

相对原版只动三处:(B) 循环外 alloc;(C) 一次性 init-SimtVF;输出行把 `w_ub[i]` 换成 `w_frag[i]`。

为什么 init 要单独包一个 SimtVF:fragment 是逐线程的 VRF,写它必须在 SimtVF 的线程映射里;而且 init-VF 给 `w_frag` 分的物理布局,必须和循环内消费 VF 逐位一致。裸写 `T.copy(w_ub, w_frag)` 让编译器自动包 init-VF 也可以,但显式形式让 lowering 一目了然。

### 3.2 TIR 形态

实跑确认:瞬时 fragment 在最终 device TIR 里是 `SIMT_VF` block 内的 `T.allocate(..., "local")`,布局由 block attr `layout_map` 指向 `tl.Fragment` 条目;MTE 和 flag 在 block 外(uniform 作用域),只有计算在 block 内,输出循环每 token 现读 `w_ub`:

```python
with T.block("SIMT_VF", no_realize=True):
    T.block_attr({"layout_map": {x_frag_2: metadata["tl.Fragment"][0], sum_sq_2: metadata["tl.Fragment"][1]}})
    x_frag = T.allocate([32], "float32", "local")
    for i in T.unroll(8):                                    # x:每 token UB→VRF(必须)
        x_frag_3[i*4:i*4+4] = x_ub_1[t%2*4096 + i*512 + simtvf_tx*4 + 4096 : ...]
    # ... Σx² + tl::AscendAllReduce + rstd ...
    for i in T.unroll(8):
        y_ub_1[...] = x_frag_3[i*4:i*4+4] * T.Broadcast(rstd_val, 4) \
                    * w_ub_1[i*512 + simtvf_tx*4 : ...]       # ← w 每 token UB→VRF(要消掉的)
```

persistent fragment 的目标 TIR 形态沿用现有 `Allocate`/`Buffer`,不新增 storage scope,只用 attr 标记生命周期(与仓内 `TileLang_RMSNorm_Register_Residency_Design.md` §3.3 对齐):

```python
# Kernel 作用域(token 循环之外):普通 local allocate + persistent attr
w_frag = T.allocate([32], "float32", "local")
T.attr(w_frag, "tl.simt_persistent", 1)

# init-VF:写 w_frag 的那个 SIMT_VF
with T.block("SIMT_VF", no_realize=True):
    T.attr("simtvf", "tl.simtvf_scope", 1)
    T.block_attr({"layout_map": {w_frag: metadata["tl.Fragment"][K]}})    # 同一 Fragment 条目 K
    for i in T.unroll(8):
        w_frag[i*4:i*4+4] = w_ub[i*512 + simtvf_tx*4 : ...]

# 循环内 consume-VF:layout_map 引用同一 K,输出改读 w_frag
with T.block("SIMT_VF", no_realize=True):
    T.block_attr({"layout_map": {x_frag: metadata["tl.Fragment"][0], ...,
                                 w_frag: metadata["tl.Fragment"][K]}})
    ...
    y_ub[...] = x_frag[...] * T.Broadcast(rstd_val, 4) * w_frag[...]       # 不再读 w_ub
```

两个 attr 承载“persistent”,都不新增 scope:buffer 级把 `tl.simt_persistent` 挂在 `local` 的 `w_frag` 上,供各 pass 识别;kernel 级用 PrimFunc attr `tl.persistent_fragments`(与 `src/transform/common/attr.h:20` 的 `tl.simt_vf_captures` 并列)汇总需要寄存器驻留的 fragment 及其 slot 区间。

哪个 SimtVF 是初始化不需要单独标记:带 `tl.simt_persistent` 的 buffer 在哪个 VF 被写(def),那个 VF 就是 init,其余读(use)的是 consume。第一版假定单次 def,即只有一个 init-VF。

### 3.3 编译器 pass 钩子(落到真实文件)

| # | 能力 | 文件 / 位置 | 要做的事 |
|---|---|---|---|
| 1 | persistent 推断 + 标注 | `tilelang/language/allocate.py:65`(`alloc_fragment`)+ 新 attr(建议 `src/transform/common/attr.h`) | 识别“alloc 支配点在所有 SimtVF 之外、却被多个 SimtVF 引用”的 fragment,打 persistent 标记,作用域为 enclosing Kernel/core(顺带解决 #29 开放问题 2 的作用域歧义)。无需新前端 kwarg。 |
| 2 | 从 per-call captures 摘除 | `src/transform/merge_shared_memory_allocations.cc:542`(计算 `tl.simt_vf_captures` 处);`src/op/simt_vf.cc` | persistent fragment 不作为每次 `asc_vf_call` 的入参(它不是要从 UB 现读的指针),改登记为整 kernel 共享的固定寄存器对象。 |
| 3 | 排除出软流水轮转 | `src/transform/pipeline_planning.cc` / `inject_pipeline.cc` / `multi_version_buffer_rewriter.cc` | `T.Pipelined(num_stages=2)` 会对 `x_ub`/`x_frag` 多版本轮转;persistent fragment 必须标为单实例,不参与 multi-versioning,不进 prologue/epilogue 复制。 |
| 4 | 跨 VF 一致 layout(element↔slot) | `src/transform/layout_inference.cc`(`local.fragment` 布局) | 保证 `w_frag` 在 init-VF 与所有 consume-VF 的 Fragment 布局逐位一致,从而每 lane 的第 k 个元素稳定对应同一 slot。物理寄存器不在这里分配,slot 编号与寄存器钉死都在 PTO 的 keep/resume(见 §4)。 |
| 5 | codegen 落地(后端相关) | PTO 路径见 §4;Ascend C 路径 `src/target/codegen_ascend.cc:489–560` | 把“persistent fragment + init/consume 结构”落成后端的寄存器驻留机制。PTO 路径生成 keep/resume(§4),不碰寄存器分配;Ascend C 路径需要对应的固定寄存器手段(本方案不展开,首版可只支持 PTO 路径)。 |
| 6 | slot/寄存器预算检查 + fallback | `tilelang/analysis/vffragment_checker.py` | keep/resume 的 slot 来自固定的 `R4..R126`(共 123 个)。要检查的是 `w 的 slot 数 + body 峰值寄存器需求 + 余量 ≤ 123`,不能只看“放得下”;否则钉住的 slot 会逼 body 把 `x_frag` 或临时量 spill 到 UB/stack,比省掉的那次 w 读更糟。超限就回退到循环内 `w_ub[i]` 现读,不允许静默 spill。对应 #29 开放问题 1,详见 §4.7。 |

### 3.4 slot / 寄存器预算与 fallback

keep/resume 的 slot 落在固定寄存器 `R4..R126`(123 个)。RMSNorm 的 consume body 同一时刻要同时持有 w、x_frag、归约临时、rstd、地址,其中 w 和 x_frag 各占 `M = TILE/threads` 个(`d=4096/threads=128` 时 M=32),合计约 70–80 个 lane 标量寄存器。把 M 个 slot 钉给 w,会压缩 BiSheng 留给 body 自身计算的预算;一旦逼出 spill,就是净负收益。预算检查见 §3.3 #6;RMSNorm 本身是这个原语的边缘 case,详见 §4.7。

---

## 4. PTO 中 IR 如何设计与对接

这一章讲 §3 的 persistent fragment 怎么落到 PTO。用的就是 PTOAS 已有的 `pto.keep` / `pto.resume`(语义见 §4.1),我们只生成这两个 op、给它们分配 slot 编号,寄存器分配仍归 BiSheng。

### 4.1 keep/resume 的语义与作用

`pto.keep` / `pto.resume` 是 PTOAS 已有的一对 op,用编译期常量 `slot` 配对,作用是让一个值在 SimtVF 调用之间驻留寄存器:

- `pto.keep %x {slot = N}`:在 `simt_entry` 内把标量 `%x` 写进 slot N(必须紧邻 `func.return`)。
- `%x = pto.resume {slot = N}`:在 `simt_entry` 内从 slot N 读回(必须是 block 第一条)。
- `slot` 直接映射到固定物理寄存器 `R4..R126`,lowering 成 inline-asm `MOV` 的 sideeffect call:`keep` 落成 `MOV R{4+N}, $0`,`resume` 落成 `MOV $0, R{4+N}`。

一个 VF 用 keep 把值写进某个 slot 后,硬件会在 VF 调用之间保留 SIMT 寄存器状态,后一个 VF 就能用 resume 从同一 slot 读回,跨 VF 的寄存器驻留就是这么实现的。因为用的是固定 `R` 寄存器加 sideeffect 的 inline asm,BiSheng 不会复用、删除或重排它,寄存器分配照旧归它(§2.2)。

一个限制:keep/resume 只接标量,禁用 `!pto.vreg` / `!pto.mask`。这不影响本方案,因为 SimtVF 里 `x_frag`/`w_frag` 是 per-lane 标量寄存器(`float4` 只是 4 个连续标量的 128-bit load),正好落在 `R4..R126` 这个标量寄存器类;被禁的是 SimdVF 的 SIMD 向量寄存器。

### 4.2 keep/resume v1 与本需求的差距

| | keep/resume v1 | RMSNorm 的 `w_frag` |
|---|---|---|
| 粒度 | 每 slot 一个标量 SSA 值 | 每 lane 一段 `M = TILE/threads` 个 f32(d=4096/threads=128 → 32) |
| 控制流 | 线性调用链,不跨分支/循环 | 跨 `n_tokens_per_core` 次迭代的 `scf.for` |
| 消费者 | 一个 slot 对一个消费者 | 一次 init 写入,被 64 次 body 反复读(只读不变量) |

这三条差距对应 §4.5 的三个扩展。

### 4.3 整体思路与新增 IR

**关键约束:每个非末尾 consumer 在 return 前都要 re-keep。** 这是最容易遗漏、却必须保证的一点,有没有循环都成立:

- RMSNorm 循环场景:`rms_body` 被外层 `scf.for` 调 64 次,某次的 body 可能把驻留 `w` 的 R4..R35 挪作他用,所以每次 return 前要把 `w` 复位,下一圈的 resume 才正确。
- 一般场景:VF0 做 init、VF1 consume、之后 VF2 也 consume,这时 VF1 的 body 同样可能动到那些寄存器,所以 VF1 必须在 return 前 re-keep,VF2 的 resume 才拿得到值。只有该 slot 的最后一个 consumer 能省(它之后没人再 resume)。

re-keep 的位置必须在 return 前,不能放在 resume 处:resume 在入口把 slot 读进 SSA 后,body 还会运行、可能覆盖这些寄存器,只有在 body 跑完、return 之前写回,下一个 consumer 进来时寄存器才是对的。所以 lowering 把 resume(入口)和 re-keep(return 前)成对生成:处理一个 consumer 的驻留读时,顺手把它的 re-keep 也安排好。第一版可以简单点,每个 consumer 都在 return 前 re-keep;末尾那条是死代码,read-only 下又是自移动,代价可忽略,优化留到 §4.5。

下面是整体思路与新增 IR。codegen 只做代码生成,不做复杂控制流变换。整条路径是:

1. 保留 TIR 的控制流(外层 token 循环、内层访存循环都不改写);
2. 把 TIR 上的 persistent 标记透传成 PTO IR 的属性;
3. unroll 内层 def/use 循环,这步省不掉,因为 keep 要钉的那 M 个值是循环体(UB load)产出的,不展开就没有;
4. 把每个 def 变成 `keep`、每个 use 变成 `resume`,再用 `keep_range`/`resume_range` 把展开后的 M 条压成一条;
5. cleanup 删掉只用于承载的脚手架。

persistent 变量在 PTO IR 上的承载放在 `pto.ptr` 的地址空间上,不用普通 local buffer 的松散 attr。这样类型不容易被 pass 丢掉,verifier 能按类型约束,def/use 也直接落到对这个指针的 store/load:

```mlir
%wfrag = pto.persistent_lane_alloc : !pto.ptr<f32, lane_persistent>   // 纯 IR 句柄,不预留寄存器
// def(init-VF 内):store → keep
pto.store %wk, %wfrag[%k] : !pto.ptr<f32, lane_persistent>            // ⇒ pto.keep %wk {slot = base+k}
// use(consume-VF 内):load → resume
%wk = pto.load %wfrag[%k] : !pto.ptr<f32, lane_persistent>            // ⇒ %wk = pto.resume {slot = base+k}
```

unroll 后 `%k` 是常量,`slot = base+k` 落定。`pto.persistent_lane_alloc` 是个纯 IR 句柄,不预留寄存器,keep/resume 生成后由 cleanup 删除(连同 simt_entry 上的 `lane_persistent` 参数,见 §4.6)。

需要的 IR 增量汇总如下:

| 项 | 新增 / 复用 | 作用 |
|---|---|---|
| `pto.keep` / `pto.resume` | 复用 PTOAS v1 | slot 与固定寄存器之间的标量搬移,lowering 成 inline-asm `MOV` |
| `lane_persistent` 地址空间(挂在 `pto.ptr` 类型上) | 新增 | 用类型标记“跨 VF 寄存器驻留的 lane-local 变量”,由步骤 2 从 TIR `tl.simt_persistent` 透传 |
| `pto.persistent_lane_alloc`(产出 `!pto.ptr<f32, lane_persistent>`) | 新增 | persistent 变量的带类型承载;纯句柄,不预留寄存器,cleanup 阶段消解 |
| `keep_range` / `resume_range` | 扩展(§4.5) | 把 unroll 后的 M 条 keep/resume 压成一条,lowering 再展开回 M 条 `MOV`;不替代 unroll |
| 多消费者中继 + verifier 放开 | 扩展(§4.5) | 让一次 keep 喂多个 consumer 成立(循环反复调用是其中一种特例) |

### 4.4 端到端示例:TIR → PTO IR → keep/resume

这里有两层循环,各管一件事,是本方案最容易混的地方:

- 外层是 token 循环(`scf.for %t`,每圈调一次 consume-VF),它决定 loop-carried 与 re-keep:`w_frag` 要在这 64 次 VF 调用之间存活。
- 内层是 `for i in T.unroll(8)`,它必须 unroll,展开后的 M 条 keep/resume 再用一条 range op 压缩。

用法是:循环前 init-VF keep 一次,循环里每个 consume-VF 入口 resume、return 前 re-keep(为什么必须 re-keep 见 §4.3)。下面按四步过一遍,省略 flag 与 DMA 细节。

**第 1 步:codegen 的输入 TIR**(persistent 版,见 §3.2)。`w_frag` 是 kernel 作用域的 `local` 加 `tl.simt_persistent`;init-VF 写它,consume-VF 读它,两处都还是 `T.unroll` 循环:

```python
@T.prim_func
def main_kernel(RSTD, W, X, Y, eps):
    T.func_attr({..., "tl.persistent_fragments": ["w_frag"]})
    w_ub   = T.decl_buffer((4096,), data=buf_dyn_shmem, scope="shared.dyn")
    w_frag = T.allocate([32], "float32", "local")             # tl.simt_persistent(kernel 作用域)
    bx = T.launch_thread("blockIdx.x", 64)
    T.ascend_copy_gm_to_ubuf(w_ub, W, ...)                    # GM->UB:W 只搬一次

    with T.block("SIMT_VF"):                                  # init-VF:def w_frag
        T.block_attr({"layout_map": {w_frag: metadata["tl.Fragment"][1]}})
        tx = T.launch_thread("threadIdx.x", 128)
        for i in T.unroll(8):
            w_frag[i*4:i*4+4] = w_ub[i*512 + tx*4 : i*512 + tx*4 + 4]

    for t in range(64):
        T.ascend_copy_gm_to_ubuf(x_ub, X[...], ...)           # MTE2
        with T.block("SIMT_VF"):                              # consume-VF:use w_frag
            T.block_attr({"layout_map": {x_frag: metadata["tl.Fragment"][0],
                                         w_frag: metadata["tl.Fragment"][1]}})
            tx = T.launch_thread("threadIdx.x", 128)
            x_frag = T.allocate([32], "float32", "local")
            for i in T.unroll(8):
                x_frag[i*4:i*4+4] = x_ub[...]
            # ... Σx² + tl::AscendAllReduce + rstd ...
            for i in T.unroll(8):
                y_ub[...] = x_frag[i*4:i*4+4] * T.Broadcast(rstd, 4) * w_frag[i*512 + tx*4 : ...]
        T.ascend_copy_ubuf_to_gm(Y[...], y_ub, ...)           # MTE3
```

**第 2 步:codegen 直接产出的 PTO IR(循环原样保留,未 unroll)**。kernel 内用 `pto.persistent_lane_alloc` 出 `%wfrag`;每个 `SIMT_VF` block 生成一个带 `{pto.simt_entry}` 的 `func.func`;对 `%wfrag` 的 store/load 还在 `scf.for` 里:

```mlir
func.func @rms_kernel(...) attributes {pto.aicore} {
  %wfrag = pto.persistent_lane_alloc : !pto.ptr<f32, lane_persistent>     // ← w_frag 的承载
  // ascend_copy_gm_to_ubuf(w_ub, W) + flags ...
  pto.simt_launch @init_wfrag<<<128,1,1>>>(%w_ub, %wfrag)                 // init-VF
  scf.for %t = 0 to 64 step 1 {
    // MTE2 + flags ...
    pto.simt_launch @rms_body<<<128,1,1>>>(%x_ub, %y_ub, ..., %wfrag, %t) // consume-VF
    // MTE3 + flags ...
  }
  return
}

func.func @init_wfrag(%w_ub: !pto.ptr<f32, ub>, %wfrag: !pto.ptr<f32, lane_persistent>)
    attributes {pto.simt_entry} {                                        // ← SIMT_VF block 生成
  %tx = pto.get_tid_x : i32
  scf.for %i = 0 to 8 step 1 {                                           // 循环还在(未 unroll)
    %v = pto.load %w_ub[%i*512 + %tx*4] : vector<4xf32>
    pto.store %v, %wfrag[%i*4] : !pto.ptr<f32, lane_persistent>          // store → persistent = def
  }
  return
}

func.func @rms_body(..., %wfrag: !pto.ptr<f32, lane_persistent>, %t: i32)
    attributes {pto.simt_entry} {
  // ... x_frag load、Σx²+all_reduce、rstd ...
  scf.for %i = 0 to 8 step 1 {                                           // 循环还在
    %w = pto.load %wfrag[%i*4] : !pto.ptr<f32, lane_persistent>          // load ← persistent = use
    // %y = %xf * %rstd * %w ; pto.store %y, %y_ub[...]
  }
  return
}
```

**第 3 步:unroll 之后,store/load 折成 keep_range/resume_range**(kernel 部分不变,只列两个 entry)。两个 `scf.for %i` 展开成 8 个 `vector<4xf32>` load,产出 `%w0..%w31`;init 的 32 条 store→keep 压成一条 `keep_range`,consume 的 32 条 load→resume 压成一条 `resume_range`,consume 出口再补一条 re-keep:

```mlir
func.func @init_wfrag(%w_ub, %wfrag) attributes {pto.simt_entry} {
  %tx = pto.get_tid_x : i32
  %w0..%w3   = pto.load %w_ub[%tx*4] : vector<4xf32>          // scf.for unroll → 8 个 float4
  // ...                                                        产出 %w0..%w31(不展开就没有)
  %w28..%w31 = pto.load %w_ub[7*512 + %tx*4] : vector<4xf32>
  pto.keep_range %wfrag, %w0, ..., %w31 {slot_base = 0, n = 32}          // 32 条 store→keep 压成一条
  return
}

func.func @rms_body(..., %wfrag, %t) attributes {pto.simt_entry} {
  %w0, ..., %w31 = pto.resume_range %wfrag {slot_base = 0, n = 32}       // 32 条 load→resume 压成一条
  // ... rstd;输出 loop(unroll)用 %wk:y = x_frag * rstd * %w{k} ...
  pto.keep_range %wfrag, %w0, ..., %w31 {slot_base = 0, n = 32}          // re-keep(consume 出口)
  return
}
```

**第 4 步:cleanup pass,删除脚手架**。keep/resume 生成后,`%wfrag` 与 `persistent_lane_alloc` 已经没有实义(slot 编号已落进 keep_range/resume_range);而且 `lane_persistent` 指针没有真实内存(就是 R4..R126),不能真跨 `simt_launch` 传,所以参数必须删掉,否则 launch ABI 落不下去。cleanup 做三件事:去掉 keep_range/resume_range 的 `%wfrag` 操作数;删 `persistent_lane_alloc`;跨函数地从 simt_entry 签名和每处 simt_launch 移除 `lane_persistent` 参数。最终形态:

```mlir
func.func @rms_kernel(...) attributes {pto.aicore} {
  // %wfrag / persistent_lane_alloc 已删除
  pto.simt_launch @init_wfrag<<<128,1,1>>>(%w_ub)                  // 不再传 %wfrag
  scf.for %t = 0 to 64 step 1 {
    pto.simt_launch @rms_body<<<128,1,1>>>(%x_ub, %y_ub, ..., %t)  // 不再传 %wfrag
  }
  return
}
func.func @init_wfrag(%w_ub) attributes {pto.simt_entry} {         // 签名去掉 %wfrag
  // ... 8 个 float4 load → %w0..%w31 ...
  pto.keep_range %w0, ..., %w31 {slot_base = 0, n = 32}            // 不再带 %wfrag
  return
}
func.func @rms_body(..., %t) attributes {pto.simt_entry} {         // 签名去掉 %wfrag
  %w0, ..., %w31 = pto.resume_range {slot_base = 0, n = 32}        // 不再带 %wfrag
  // ... 用 %wk ...
  pto.keep_range %w0, ..., %w31 {slot_base = 0, n = 32}            // re-keep
  return
}
```

cleanup 之后,keep_range/resume_range 只剩 slot 属性,lower 成定址 `R4..R126` 的 `MOV`,persistent 指针在 IR 里不留痕迹;verifier 改成按 slot(而非 `%wfrag`)匹配 keep/resume,和 PTOAS v1 一致。`init_wfrag` 与每次 `rms_body` 都引用同一组 slot(0..31),在 call 链上形成 `keep → (resume…re-keep)×64` 的循环携带数据流,中间不再从 UB 重载 w。

### 4.5 对 keep/resume v1 的扩展

1. **`keep_range` / `resume_range`**:把 unroll 后的 M 条单 slot keep/resume 压成一条带 `{slot_base, n=M}` 的 range op(IR 32→1),lowering 时再展开回 M 条 `MOV R{4+base+k}`。它只减少 IR 体积,不替代 unroll,要钉的那 M 个值仍由展开的循环体产出。
2. **多消费者中继 + verifier 放开**:一次 init 的 keep 可以喂多个 consumer,每个 consumer 按“resume 入口、re-keep 返回”把值接力给下一个,形成 `keep(init) → [resume+rekeep] → [resume+rekeep] → …` 的链。判定与循环无关:只要某个 consumer 之后还有 VF 会 resume 同一 slot,它就要 re-keep,只有该 slot 的最后一个消费者能省。RMSNorm 是其中的循环特例:consumer 是同一个 `rms_body` 被 `scf.for` 调 64 次。这要放开 PTOAS v1 的三条约束(`vpto-simt-keep-resume-design.md` §5.3 / §5.4):一个 slot 只对一个消费者、第一版只支持线性链不跨循环、keep 与 resume 之间不得有新的 simt_entry 调用,这三条多消费者中继都要松开。放开后换上的检查:每个非末尾 consumer 的 resume 入口与 re-keep 出口成对、init-keep 支配所有 consumer、launch 维度一致。keep/resume 的 sideeffect(`Write/Read<SLOT>`)天然挡住 pass 的重排与复制。
3. **read-only 标注(可选)**:标 `{readonly}` 让 lowering 知道 re-keep 是自移动、可整段消除。第一版可以不做,直接 re-keep,正确性不依赖它。

### 4.6 lowering pass:`pto-lower-persistent-fragment`

识别 `tl.simt_persistent` 的 `lane_persistent` 指针,按 def/use 把 VF 分两类:写它的 SIMT_VF 是 init,读它的是 consume。lowering 规则(纯翻译,不做寄存器分配):

1. **内层 unroll**:def/use 该指针的内层循环先展开,产出 M 个值和 M 个 use 站点;不展开就没有值可 keep。
2. **def → keep,use → resume**:每个 def 一条 `keep`、每个 use 一条 `resume`;consume 侧把 `w_frag[k]` 的读 rebind 到 resume 的产出,删掉 persistent load。
3. **用 range op 压缩**:把 M 条 keep/resume 各压成一条 `keep_range`/`resume_range`,lowering 再展开回 M 条 `MOV`。
4. **非末尾 consumer 加 re-keep**:一个 consumer VF 只要之后还有 VF 会 resume 同一 slot,就在 `return` 前补一条 `keep_range` 把值接力下去;只有该 slot 的最后一个消费者能省(与有没有循环无关)。RMSNorm 里表现为 `rms_body` 每次 return 都 re-keep、末次迭代那条是死代码;init-VF 不读 slot,只 keep 不 re-keep。
5. **slot 编号**:`slot_base` 从 `R4` 起按 fragment 顺序排,这只是编号,不是物理寄存器分配。
6. **cleanup**:删 `persistent_lane_alloc`,去掉 range op 的 `%wfrag` 操作数,跨函数从 simt_entry 签名和每处 simt_launch 移除 `lane_persistent` 参数(见 §4.4 第 4 步)。
7. **预算回退**:与前端 §3.3 #6 联动,slot 不足就退回 UB 现读。

这些规则对应 §4.4 示例的第 2、3、4 步:规则 1–4 完成第 2 步到第 3 步(展开、生成 keep/resume、压缩、re-keep),规则 6 是第 4 步的 cleanup,规则 5 和 7 分别管 slot 编号与预算回退。

verifier 在 PTOAS v1 基础上扩展:keep/resume 仅在 simt_entry 内;slot 区间映射到 `R4..R126` 且不越界;init 的 keep_range 支配外层循环;init 与 consume 的 launch 维度一致;同一 fragment 的 element↔slot↔UB-offset 映射两端一致。

### 4.7 关键风险:寄存器压力

`R4..R126` 共 123 个 slot。RMSNorm 的 consume body 同一时刻要持有 w、x_frag、归约临时、rstd、地址,`d=4096/threads=128` 时约 70–80 个 lane 标量寄存器。把 M 个 slot 钉给 w,会压缩 BiSheng 留给 body 的预算;一旦逼出 spill,比省掉的那次 w 的 UB 读更糟,是净负收益。所以预算检查不能只看“M ≤ 123”,要算上 body 自己的峰值:`w 的 slot 数 + body 峰值需求 + 余量 ≤ 123`,否则回退。由此有三点:

- RMSNorm 是这个原语的边缘 case:x_frag 本就占掉半个寄存器堆,再钉 32 个给 w 压力很大,而且它 HBM-bound、wall-clock 大概率打平。
- RoPE 的 cos/sin、LayerNorm/GroupNorm 的 bias 是更好的首发载体:不变量更小(占 slot 少)、每 token 复用多,不易 spill,收益也更实。
- 放不下时可以部分驻留(只 keep 一半 w、另一半仍从 UB 读),按 slot 预算定比例;第一版先只做“全驻留或全回退”。

### 4.8 与 `ptoas` / `ptodsl` 的关系

keep/resume 本就是 PTO MLIR 的 op,`ptoas` 直接走它的 inline-asm lowering;`ptodsl` 只需把它 pretty-print 成 PTODSL 源,不承载独立语义。本方案不新增并列 IR,只在 PTOAS keep/resume 上加 §4.5 的三个扩展。

### 4.9 TIR → PTO 映射增量(相对基础稿 §3.5)

| TIR 构造(本方案新增) | PTO IR | 说明 |
|---|---|---|
| `w_frag` allocate(`local` + `tl.simt_persistent` attr) | slot 区间 `[base, base+M)` | aicore 作用域,跨 launch 寄存器驻留;不新增 scope/op |
| init-`SIMT_VF` block 末尾 | `pto.keep_range {slot_base, n}`(紧邻 return) | 循环外只发一次 |
| consume-`SIMT_VF` 入口 / 返回前 | `pto.resume_range`(block 首)+ `pto.keep_range`(return 前) | 入口取回、返回前复位;输出用 `%w[i]`,不再 `UB→VRF` |
| PrimFunc attr `tl.persistent_fragments` | `pto-lower-persistent-fragment` 的输入 | 驱动 slot 分配、生成 keep/resume、预算回退 |
| `x_frag` allocate(`local.fragment`) | `simt_entry` 内本地寄存器数组 | 瞬时,不变 |

---

## 5. 落地步骤与开放问题

### 5.1 建议落地顺序

1. 先验证底座:用最小用例确认 PTOAS 的 keep/resume 跨调用语义(两个相邻 simt_entry 一存一取,值不变),以及 §4.5 的多消费者中继扩展(`scf.for` 内 resume + re-keep 跨迭代正确,以及线性多 consumer)。
2. PTO 侧扩展:在 PTOAS keep/resume 上加 `keep_range`/`resume_range`、多消费者中继(resume + re-keep),以及可选的 read-only,并扩 verifier。
3. 前端/TIR:#1 persistent 推断、#4 一致 layout、#2 摘 captures、#3 排除轮转、#6 slot 预算与 fallback;TIR→PTO 走 `pto-lower-persistent-fragment`(§4.6)。
4. 首发选 RoPE,不选 RMSNorm:按 §4.7,RMSNorm 寄存器压力大且 HBM-bound;先在 RoPE 的 cos/sin(或 LayerNorm bias)上跑通端到端并校验。
5. 先量后改:profiling 确认瓶颈是 HBM 还是 UB 端口或寄存器压力,据此决定推广面。

### 5.2 测试分层

- 数值:对照参考实现,`diff < 1e-3`。
- IR 级:device TIR 中 `w_frag` 提到循环外、init/consume 共享同一 `tl.Fragment` 条目;PTO IR 中 init-VF 有 `keep_range`、consume-VF 有 `resume_range` 加 `keep_range`,slot 区间一致。
- LLVM/asm 级:consume body 的 `resume_range` 落成 `MOV $0, R{4+k}` 的 sideeffect inline asm;输出不再有 w 的 `UB→VRF` 现读。
- 预算/fallback:构造 `M + body 峰值 > 123` 的配置,确认它回退到 UB 现读,没有钉 slot 逼出 spill。

### 5.3 开放问题

1. **slot/寄存器预算**:`R4..R126` 共 123 个;`M + body 峰值需求 + 余量 ≤ 123` 的阈值,以及“全驻留 / 部分驻留 / 全回退”的策略还需定(§4.7)。
2. **re-keep 开销**:每迭代 re-keep M 个 slot,理想情况是自移动被后端消除;需要实测确认 BiSheng 是否真的 elide,否则考虑 read-only 区间保留以省掉 re-keep。
3. **多消费者中继 verifier**:放开 v1 的“线性链 / 单消费者”后,需要测例覆盖多 consumer(循环与线性链两种)、重排、复制、分支的安全性。
4. **驻留作用域**:用“alloc 在 SimtVF 外即 enclosing Kernel 作用域”消解 #29 的歧义;一个 kernel 内多组 SimtVF 或多个寄存器驻留的 fragment 之间的交互还需测例。
5. **收益面**:RMSNorm 大概率打平且压力大,首发用 RoPE/bias 验证真实收益。

---

## 附:与基础稿的关系

本方案是基础稿 [`TileLang_RMSNorm_to_PTO_Design.md`](https://github.com/KurrinQu/PTOAS/blob/tilelang-rmsnorm-pto-design-doc/docs/designs/TileLang_RMSNorm_to_PTO_Design.md) 的增量。基础稿解决“RMSNorm 怎么落到 PTO”(simt_launch/entry、float4 向量访存、AscendAllReduce helper),本方案在其上加“循环不变量 W 如何跨 SimtVF 调用做寄存器驻留”,直接复用 PTOAS 的 `pto.keep` / `pto.resume`,只加 §4.5 的三个扩展。两者取向一致:TileLang 侧生成薄语义,寄存器分配与底层实现交给 BiSheng / PTO。
