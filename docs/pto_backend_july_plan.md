# PTO 后端 7 月工作计划

## 整体目标

7 月 PTO 后端建设重点为 SIMT 和 CUBE/GEMM example。目标是先用 RMSNorm 和 GEMM 打通两条主线，再在 7 月底前补齐其余 SIMT / CUBE 类 example，并合入主干。

| 时间点 | 目标 | 验收标准 |
|---|---|---|
| 7.15 | 打通 `example_rmsnorm.py` / `example_gemm.py` 并合入主干 | RMSNorm 和 GEMM 在 PTO target 下 compile + run 通过；相关代码、CI 用例合入主干 |
| 7.30 | 打通其余 SIMT / CUBE 类 example，并完成 SIMT 寄存器驻留优化后合入主干 | 剩余 SIMT / CUBE example 在 PTO target 下 compile + run 通过；SIMT 寄存器驻留优化完成 RMSNorm 类性能优化闭环；完成合入和回归验证 |

## SIMT example 分类与排期

`example_rmsnorm.py` 作为 7.10 主线目标，其它 SIMT pipeline / sync / FP8 项作为 7.15 和 7.30 的扩展验证目标。

| Example 分类 | example 名称 | 目标时间 | 责任人 |
|---|---|---:|---|
| RMSNorm 主线 | `example_rmsnorm.py` | 7.10 | zhangyuchen |
| SIMT Pipeline 扩展 | `example_simtvf_vecadd.py`<br>`example_simtvf_vector_add.py`<br>`example_buffer_version_annotation.py` | 7.15 | zhangyuchen |
| SIMT 多 kernel | `example_simtvf_multi_kernel.py` | 7.15 | baitian |
| SIMT UB / Auto Sync | `example_simtvf_ubuf_multi.py`<br>`example_simtvf_auto_sync.py` | 7.15 | baitian |
| Mutex / Buffer Token Pipeline | `example_simtvf_vecadd_mutex.py` | 7.30 | baitian |
| SIMT 动态 stride / Vectorized Load | `example_int64_stride_vectorize_load.py` | 7.30 | baitian |
| SIMT Atomic Add | `example_atomic_add.py` | 7.30 | zhangyuchen |
| FP8 / Per-token Quantize | `example_simtvf_per_token_cast_to_fp8.py` | 7.30 | zhangyuchen |

## CUBE / GEMM example 分类与排期

CUBE / GEMM 类 example 按是否和 `example_gemm.py` 主线能力直接相关拆分。`example_gemm.py` 是 7.10 CUBE 主线目标，覆盖基础 L1/L0C 分配、GM->L1 copy、`T.gemm`、`transpose_B`、`clear_accum`、`unit_flag_ctrl`、HF32、FP8/BF16/FP32 dtype 和 L0C 输出。其余 example 按新增能力继续拆分到 7.15 联动验证或 7.30 剩余目标。

| Example 分类 | example 名称 | 主要特性 / PTO 关注点 | 目标时间 | 责任人 |
|---|---|---|---:|---|
| GEMM 主线 | `example_gemm.py` | auto-scheduled GEMM 主线；<br>`T.Persistent`、`T.Pipelined`；<br>`alloc_l1`、`alloc_l0c`；<br>GM->L1 copy、W transpose copy；<br>`T.gemm(..., transpose_B=True, clear_accum=..., unit_flag_ctrl=...)`；<br>BF16 / FP32 / FP8 dtype、BF16 out、HF32 mode；<br>`T.dual_copy` 或 L0C->GM 输出；<br>作为 CUBE/GEMM 主线目标 | 7.10 | yangben |
| L0 显式搬运 | `example_gemm_l0.py` | 新增 `alloc_l0a` / `alloc_l0b`；<br>覆盖 L1->L0A/L0B copy；<br>嵌套 K-sub tile pipeline；<br>重点是 L0A/L0B/L0C memory scope 与 `T.gemm` codegen | 7.15 | yangben |
| MixedKernel / Manual Mix 联动项 | `example_gemm_mixedkernel.py`<br>`example_gemm_mix_manual.py` | `T.MixedKernel`；<br>`T.Cube()` / `T.Vector()` 手写分支；<br>AIC/AIV mixed ABI、subblock id；<br>cross-core flag；<br>L0C->UB->GM split 输出；<br>验证 mixed kernel launch metadata 和同步协议 | 7.30 | yangben |
| Copy / L2 / Shape 特性 | `example_gemm_bypass_l2.py`<br>`example_gemm_various_shapes.py`<br>`example_annotate_unlimit_memory.py` | `l2_cache_ctrl` 参数；<br>L2 bypass / not-alloc copy 策略；<br>多 shape、多 tile swizzle；<br>Half-N split、双 L0C accumulator、双输出；<br>`T.annotate_buffer_versions` 和 `T.annotate_unlimit_memory`；<br>重点是 copy lowering 参数、cache control、复杂 shape 和 buffer version 传递 | 7.30 | yangben |
| Blockscaled / MX GEMM | `example_blockscaled_gemm.py`<br>`example_blockscaled_gemm_l0.py` | `T.blockscaled_gemm`；<br>scale factor L1 copy、`sf_k_within_chunk`；<br>scale-aware L1->L0 copy；<br>MXFP8 GEMM 变体 intrinsic；<br>L1 版本和 L0 版本都需要验证 | 7.30 | yangben |

## 优化项

| 优化方向 | 关联问题 / 用例 | 主要目标 | 目标时间 | 责任人 |
|---|---|---|---:|---|
| SIMT 寄存器驻留优化 | issue #29: `SimtVF` 支持跨调用保活的 fragment / 寄存器驻留 | 暴露跨 VF 保活 fragment 语义，避免 token-wise 场景中循环不变量重复 UB -> VRF 访问；<br>补齐 keep/resume 相关 codegen 与 layout/slot 传递；<br>优先作用于 RMSNorm / LayerNorm / RoPE 等 token-wise 融合 kernel；<br>目标是形成 RMSNorm 类 token-wise kernel 的性能优化闭环并合入主干 | 7.30 | qukelin |
