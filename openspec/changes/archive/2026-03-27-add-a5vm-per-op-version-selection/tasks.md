## 1. IR 契约与 pass 接线

- [x] 1.1 在 `include/PTO/IR/` 中新增 `pto.a5vm_lowering_choice` typed attribute 及其 `update_mode`、`loop_shape` 枚举定义，并补齐相应生成代码接线。
- [x] 1.2 在 `include/PTO/Transforms/Passes.td`、`include/PTO/Transforms/Passes.h` 和对应实现中声明 `PTOA5VMVersionSelectionPass`，并为后续 backend cleanup 阶段预留接线位置。
- [x] 1.3 在共享 lowering 头文件中新增 per-op 版本选择读取与校验 helper，统一 `PTOA5VMVersionSelectionPass` 和 `PTOToA5VM` 的属性访问口径。

## 2. 版本选择 pass

- [x] 2.1 在 `lib/PTO/Transforms/` 实现 `PTOA5VMVersionSelectionPass`，遍历 `func::FuncOp` 内全部 A5VM candidate PTO op，包括 `pto.fusion_region` 内外两类 op。
- [x] 2.2 将现有 `PTOToA5VMLowering.cpp` 中分散的 `post_update/no_post_update` 与 `1D/2D` 判定逻辑抽取为可复用决策函数，供版本选择 pass 复用。
- [x] 2.3 让版本选择 pass 为每个目标 op 写入完整的 `pto.a5vm_lowering_choice`，选择逻辑是对fusion_region内OP全部选择2D + no_post_update, region外OP选择2D + post_update，并对无法判定的 case 给出明确诊断。

## 3. `PTOToA5VM` direct-emission 改造

- [x] 3.1 修改 `lib/PTO/Transforms/PTOToA5VM.cpp`，在 fusion mainline 中把 per-op `pto.a5vm_lowering_choice` 作为唯一正式决策输入。
- [x] 3.2 修改 `lib/PTO/Transforms/PTOToA5VMLowering.cpp`，按 `update_mode` 与 `loop_shape` 直接 dispatch 到选定版本，去掉以版本选择为目的的主 `if/else` 生成。
- [x] 3.3 对缺失属性、非法属性和不兼容组合实现 fail-fast 诊断，保证错误在 `PTOToA5VM` 边界暴露。
- [x] 3.4 保持非 fusion 路径现有 `--a5vm-lowering-strategy` 行为不变，并在 fusion mainline 下对该参数输出 ignored warning。

## 4. Cleanup pass 与 backend pipeline

- [x] 4.1 不新增 `PTOA5VMConstIfCleanupPass`，改为在 `tools/ptoas/ptoas.cpp` 中直接调用公共 `Canonicalizer` 作为 cleanup 阶段。
- [x] 4.2 调整 `tools/ptoas/ptoas.cpp` 中的 A5 fusion mainline 顺序为 `PTOFusionRegionGen -> PTOA5VMVersionSelection -> PTOToA5VM -> Canonicalizer -> PTOLowLevelLoopFusion -> CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion -> CSE`。
- [x] 4.3 检查 `PTOLowLevelLoopFusion`、`PTOFusionPredicateElision`、`PTOFusionLoadStoreElision` 和 `PTOFlattenFusionRegion` 对新 per-op 属性与 cleanup 结果的输入契约未被破坏。

## 5. 回归与验证

- [x] 5.1 新增或更新 lit 测试，验证 `PTOA5VMVersionSelectionPass` 会为 `pto.fusion_region` 内外全部目标 op 标注 `pto.a5vm_lowering_choice`。
- [x] 5.2 新增或更新 lit 测试，验证 `PTOToA5VM` 对已标注 op 直接生成选定版本，且不再输出用于版本选择的 `scf.if` / `else`。
- [x] 5.3 新增负例测试，验证缺失属性、非法枚举值和不兼容组合会导致 `PTOToA5VM` fail-fast。
- [x] 5.4 新增 cleanup 回归，验证 `Canonicalizer` 会删除常量条件分支且保守 case 不被误折叠。
- [x] 5.5 运行最小相关验证，包括至少 1 个 fusion mainline 集成用例和 1 个非 fusion `--a5vm-lowering-strategy` 兼容用例，并记录结果。

## 验证记录

- 2026-03-26：`/home/zhangzhendong/ptoas-workspace/llvm-project/build-shared/bin/llvm-lit -sv test/phase2/a5vm_fusion_pipeline_order.mlir test/phase2/a5vm_version_selection_annotations.mlir test/phase2/a5vm_direct_emission.mlir test/phase2/a5vm_fail_fast_missing_attr.mlir test/phase2/a5vm_fail_fast_bad_attr_type.mlir test/phase2/a5vm_fail_fast_bad_choice_combo.mlir test/phase2/a5vm_canonicalizer_cleanup.mlir test/phase2/a5vm_fusion_region_lifecycle.mlir test/phase2/a5vm_fusion_predicate_elision.mlir`
- 2026-03-26：`source scripts/ptoas_env.sh && build/tools/ptoas/ptoas test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --a5vm-lowering-strategy=no-post-update --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null`
  结果：成功到 `IR Dump After PTOFlattenFusionRegion`，并输出 `Warning: --a5vm-lowering-strategy is ignored because A5 fusion mainline uses per-op pto.a5vm_lowering_choice.`
- 2026-03-26：`source scripts/ptoas_env.sh && build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-lowering-strategy=no-post-update --a5vm-print-ir test/phase2/pto_backend_a5vm_wiring.mlir -o /dev/null`
  结果：成功输出 `A5VM IR op: a5vm.copy_ubuf_to_gm` 等 A5VM IR，未出现 ignored warning，确认非 fusion 兼容路径仍走全局策略接口。
