# PTO 后端 7 月工作计划

## 整体目标

7 月 PTO 后端建设重点为 SIMT 和 CUBE/GEMM example。目标是先用 RMSNorm 和 GEMM 打通两条主线，再在 7 月底前补齐其余 SIMT / CUBE 类 example，并合入主干。

| 时间点 | 目标 | 验收标准 |
|---|---|---|
| 7.15 | 打通 `example_rmsnorm.py` / `example_gemm.py` 并合入主干 | RMSNorm 和 GEMM 在 PTO target 下 compile + run 通过；相关代码、CI 用例合入主干 |
| 7.30 | 打通其余 SIMT / CUBE 类 example，并完成 SIMT 寄存器驻留优化后合入主干 | 剩余 SIMT / CUBE example 在 PTO target 下 compile + run 通过；SIMT 寄存器驻留优化完成 RMSNorm 类性能优化闭环；完成合入和回归验证 |

## SIMT example 分类与排期

`example_rmsnorm.py` 和 `example_rmsnorm_pto.py` 视为同一个 RMSNorm example，二者 kernel 逻辑一致，差异仅是 target 从 `ascend` 切换为 `pto`。`example_rmsnorm.py` 作为 7.10 主线目标，其它 RMSNorm / VecAdd pipeline 联动项作为 7.15 缓冲验证目标。

| Example 分类 | example 名称 | 目标时间 | 责任人 |
|---|---|---:|---|
| RMSNorm 主线 | `example_rmsnorm.py` / `example_rmsnorm_pto.py`<br>同一 kernel，仅 target 从 `ascend` 切到 `pto` | 7.10 | zhangyuchen |
| SIMT Pipeline 扩展 | `example_rmsnorm_auto.py`<br>`example_simtvf_vecadd.py`<br>`example_simtvf_vecadd_auto.py`<br>`example_simtvf_vector_add.py`<br>`example_buffer_version_annotation.py` | 7.15 | zhangyuchen |
| SIMT 多 VF 与 UB 同步 | `example_simtvf_ubuf.py`<br>`example_simtvf_ubuf_multi.py`<br>`example_simtvf_copy.py`<br>`example_simtvf_auto_sync.py` | 7.15 | baitian |
| Mutex / Buffer Token Pipeline | `example_simtvf_vecadd_mutex.py` | 7.30 | baitian |
| FP8 / Per-token Quantize | `example_simtvf_per_token_cast_to_fp8.py` | 7.30 | zhangyuchen |

## CUBE / GEMM example 分类与排期

CUBE / GEMM 类 example 按是否和 `example_gemm.py` 主线能力直接相关拆分。`example_gemm.py` 是 7.15 CUBE 主线目标，覆盖基础 L1/L0C 分配、GM->L1 copy、`T.gemm`、`transpose_B`、`clear_accum`、手写 flag pipeline 和 L0C->GM 输出。其余 example 按新增能力继续拆分到 7.15 联动验证或 7.30 剩余目标。

| Example 分类 | example 名称 | 主要特性 / PTO 关注点 | 目标时间 | 责任人 |
|---|---|---|---:|---|
| GEMM 主线 | `example_gemm.py` | 手写 CUBE pipeline；<br>`alloc_l1`、`alloc_l0c`；<br>GM->L1 copy、W transpose copy；<br>`T.gemm(..., transpose_B=True, clear_accum=...)`；<br>L0C->GM 输出；<br>作为 CUBE/GEMM 主线目标 | 7.10 | yangben |
| GEMM auto pipeline 联动项 | `example_gemm_auto.py` | 和 GEMM 主线计算路径接近；<br>新增 `T.Persistent`、`T.Pipelined`；<br>新增 `T.dual_copy` 输出路径；<br>覆盖 bf16 / fp32 / fp8 dtype 入口、HF32 mode；<br>主要验证 PTO 是否复用 Ascend auto pipeline lowering 和 launch metadata | 7.15 | yangben |
| L0 显式搬运 | `example_gemm_l0.py`<br>`example_gemm_l0_auto.py` | 新增 `alloc_l0a` / `alloc_l0b`；<br>覆盖 L1->L0A/L0B copy；<br>嵌套 K-sub tile pipeline；<br>手写 flag 版和 auto pipeline 版都需要验证；<br>重点是 L0A/L0B/L0C memory scope 与 `T.gemm` codegen | 7.30 | yangben |
| Copy / L2 / Padded 特性 | `example_gemm_padded_copy.py`<br>`example_gemm_bypass_l2.py` | padded GM->L1 transpose copy；<br>非完整 N tile / padded N_pad 输出；<br>`l2_cache_ctrl` 参数；<br>L2 bypass / not-alloc copy 策略；<br>重点是 copy lowering 参数、边界 shape 和 cache control 透传 | 7.30 | yangben |
| Shape / Split / Blockscaled | `example_gemm_various_shapes.py`<br>`example_blockscaled_gemm_auto.py` | 多 shape、多 tile swizzle；<br>Half-N split、双 L0C accumulator、双输出；<br>`T.annotate_buffer_versions` + auto pipeline；<br>`T.blockscaled_gemm`、scale factor L1 copy、`sf_k_within_chunk`；<br>重点是 GEMM 变体 intrinsic 和复杂 buffer/layout 传递 | 7.30 | yangben |

## 优化项

| 优化方向 | 关联问题 / 用例 | 主要目标 | 目标时间 | 责任人 |
|---|---|---|---:|---|
| SIMT 寄存器驻留优化 | issue #29: `SimtVF` 支持跨调用保活的 fragment / 寄存器驻留 | 暴露跨 VF 保活 fragment 语义，避免 token-wise 场景中循环不变量重复 UB -> VRF 访问；<br>补齐 keep/resume 相关 codegen 与 layout/slot 传递；<br>优先作用于 RMSNorm / LayerNorm / RoPE 等 token-wise 融合 kernel；<br>目标是形成 RMSNorm 类 token-wise kernel 的性能优化闭环并合入主干 | 7.30 | qukelin |

