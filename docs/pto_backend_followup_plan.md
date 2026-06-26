# PTO 后端后续完善计划

## 总体方向

当前 PTO 后端已经完成基础接入：
- `target="pto"` 可以复用 Ascend lowering pipeline，并在最终 codegen 阶段选择 PTO PTODSL codegen。
- 当前已经跑通基础 `example_simdvf_vecadd_lower_pto.py` 用例，后续目标是以 `examples/ascend` 中的 SIMD、SIMT、CUBE 三类 example 为验收入口，逐步完善 PTO 后端能力。

| 方向 | 当前状态 | 后续建设目标 |
|---|---|---|
| PTODSL Codegen | 已支持基础 SimdVF vecadd lower | 逐步覆盖 SIMD intrinsic、SIMT fragment/reduce、CUBE GEMM intrinsic |
| Example 覆盖 | 当前 1 个基础 PTO smoke 用例 | 按 SIMD -> SIMT -> CUBE -> MIX融合逐步扩展 |

## 工作计划

计划完成时间：7.30

| 工作方向 | 代表用例 | 当前基础 | 后续要补齐的能力 | 验收目标 | 需要的人力投入 |
|---|---|---|---|---|---|
| SIMD 基础与复杂控制流 | `example_simdvf_vecadd.py`、`example_simdvf_per_token_cast_to_fp8.py`、`example_simdvf_topk_gate.py`、`example_simdvf_scalar_topk.py`、`example_simdvf_scalar_topk_scalar_write.py`、`example_simdvf_vintlv.py` | 已支持 `example_simdvf_vecadd_lower_pto.py` 基础对接，包括 GM/UB copy、UB allocation、部分 SIMD intrinsic、基础 multibuffer/pipeline sync | 补齐其余 SIMD 指令的 PTODSL codegen；支持 mask、cast、比较表达式、`if/for`、标量变量、UB/GM 标量读写；扩展并验证复杂 multibuffer/pipeline sync 模式 | SIMD vecadd、cast、topk、vintlv 类 example 通过 | 1人月 |
| SIMT / Fragment / Reduce / Math | `example_simtvf_vector_add.py`、`example_simtvf_vecadd.py`、`example_simtvf_vecadd_mutex.py`、`example_simtvf_ubuf_multi.py`、`example_simtvf_auto_sync.py`、`example_simtvf_per_token_cast_to_fp8.py`、`example_rmsnorm.py`、`example_buffer_version_annotation.py`、`example_int64_stride_vectorize_load.py`、`example_atomic_add.py` | 已复用 Ascend pipeline，但 PTO codegen 对 SIMT/fragment/reduce 未覆盖 | 支持 `T.SimtVF`、`T.Parallel`、thread id、fragment/local buffer、UB <-> fragment copy；支持 `reduce_sum/max/absmax`、`exp/sqrt/rsqrt/abs`、fp8/bf16/fp16/fp32 cast、SIMT 自动同步语义、atomic；补齐 SIMT launch 对 `dynamic UB size` 的传参与元信息 | SIMT vector add、ubuf multi、fp8 cast、rmsnorm、atomic 类 example 通过 | 2人月 |
| CUBE / GEMM 基础 | `example_gemm.py`、`example_gemm_l0.py`、`example_gemm_bypass_l2.py`、`example_gemm_various_shapes.py` | PTO 目前尚未覆盖 CUBE memory scope 和 GEMM intrinsic | 支持 `alloc_l1/l0a/l0b/l0c`；支持 GM->L1、L1->L0、L0C->UB/GM；支持 `tl.ascend_mad`、`tl.ascend_gemm_l1`、`transpose_B`、`clear_accum`、`hf32`、`unit_flag_ctrl` | 普通 GEMM、L0 GEMM、多 shape GEMM example 通过 | 1人月 |
| CUBE 高级与 AIC/AIV 协同 | `example_blockscaled_gemm.py`、`example_blockscaled_gemm_l0.py`、`example_gemm_mix_manual.py`、`example_gemm_mixedkernel.py`、`example_annotate_unlimit_memory.py` | 已完成 PTO 作为 Ascend codegen backend 的结构接入，但 mixed kernel/Cube-Vector 协同还未覆盖 | 支持 blockscaled GEMM、scale factor 搬运、`dual_copy`、`nd2nz`、`T.MixedKernel`、`T.Cube`、`T.Vector`、AIC/AIV sid、unlimit memory annotation；补齐 mixed kernel launch 的 runtime 元信息和参数传递 | blockscaled GEMM、manual mixed GEMM、mixed kernel example 通过 | 1人月 |
| CV融合用例 | `example_flash_attn.py` | 依赖 SIMD、CUBE、pipeline 和 launch/runtime 元信息等多项能力，目前适合作为后期综合验收 | 联合验证 L1/L0/UB 搬运、GEMM、`dual_copy`、`nd2nz`、SIMD softmax/update、复杂 pipeline、多 stage、多 buffer | FlashAttention example 通过 | 1人月 |
| SIMT 优化项 | issue #29: `SimtVF` 支持跨调用保活的 fragment / 寄存器驻留 | 目前默认按每次进入 `SimtVF` 重新加载寄存器数据处理 | 暴露跨 VF 保活 fragment 语义，避免 token-wise 场景中循环不变量重复 UB -> VRF 访问；补齐 keep/resume 相关 codegen 与 layout/slot 传递 | RMSNorm / LayerNorm / RoPE 等 token-wise 融合 kernel 相关优化落地 | 1人月 |

## Example 覆盖清单

| 分类 | 当前纳入规划的 example |
|---|---|
| SIMD | 已对接基线：`example_simdvf_vecadd_lower_pto.py`；后续扩展：`example_simdvf_vecadd.py`、`example_simdvf_per_token_cast_to_fp8.py`、`example_simdvf_topk_gate.py`、`example_simdvf_scalar_topk.py`、`example_simdvf_scalar_topk_scalar_write.py`、`example_simdvf_vintlv.py` |
| SIMT | `example_simtvf_vector_add.py`、`example_simtvf_vecadd.py`、`example_simtvf_vecadd_mutex.py`、`example_simtvf_ubuf_multi.py`、`example_simtvf_auto_sync.py`、`example_simtvf_per_token_cast_to_fp8.py`、`example_rmsnorm.py`、`example_buffer_version_annotation.py`、`example_int64_stride_vectorize_load.py`、`example_atomic_add.py`、`example_ascend_postproc_callback.py` |
| CUBE / GEMM | `example_gemm.py`、`example_gemm_l0.py`、`example_gemm_bypass_l2.py`、`example_gemm_various_shapes.py`、`example_blockscaled_gemm.py`、`example_blockscaled_gemm_l0.py`、`example_gemm_mix_manual.py`、`example_gemm_mixedkernel.py`、`example_annotate_unlimit_memory.py` |
| 综合融合 | `example_flash_attn.py` |
| 辅助特性 | `example_make_tensor_ptr.py` |

## 其它

1. 7.15目标为SIMTVF合入，依赖DSL SIMT相关特性，存在一定风险。
2. 7.30目标为SIMD对接，主要工作为通过VMI对接，这部分方案和计划尚未评估，需要对齐方案和计划。
3. tile-kernel对pto后端相关需求尚未系统性梳理，问题触发，存在一定风险。
4. PTO后端合入DS后，后续用户CANN包升级，需要PTOAS适配，存在版本升级适配的工作量，PTOAS和DSL的版本发布问题。