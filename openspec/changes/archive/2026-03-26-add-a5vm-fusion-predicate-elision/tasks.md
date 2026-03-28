## 1. Pass 接线与主线顺序

- [x] 1.1 在 `include/PTO/Transforms/Passes.h` 与 `include/PTO/Transforms/Passes.td` 声明并注册 `PTOFusionPredicateElisionPass`。
- [x] 1.2 在 `lib/PTO/Transforms/TileFusion/` 新增 pass 实现文件并接入构建。
- [x] 1.3 在 `tools/ptoas/ptoas.cpp` 将新 pass 固定插入 A5 fusion 主线内部 `CSE` 与 `PTOFusionLoadStoreElisionPass` 之间，且不影响 non-fused path。

## 2. Fusion-region-local predicate-elision 实现

- [x] 2.1 仅遍历 `pto.fusion_region` body 内的 `a5vm.plt_b8`、`a5vm.plt_b16`、`a5vm.plt_b32`，建立候选收集与支配关系检查。
- [x] 2.2 实现整 op 双结果复用：当后一个 `plt` 的 scalar 输入与前一个支配 `plt` 等价时，重写 `mask` 与 `scalar_out` 全部 uses，并删除冗余 op。
- [x] 2.3 实现支持 `scf.for` iter_args 的 loop-carried scalar 等价传播，仅在 init 等价且 `scf.yield` 由等价 `plt.scalar_out` 递推时继续传播。
- [x] 2.4 对不同 bitwidth、无支配关系、recurrence 分叉或超出支持 IR 形态的场景保持保守不变。

## 3. 回归与验证

- [x] 3.1 新增或更新 `test/phase2` 用例，检查 `PTOFusionPredicateElisionPass` 之后 fusion region 内重复 `plt_b32` 被折叠，且 pass 顺序位于 `CSE` 与 `PTOFusionLoadStoreElisionPass` 之间。
- [x] 3.2 新增 `test/phase2` family 覆盖与负例，验证 `plt_b8` / `plt_b16` / `plt_b32` 共享同一规则，而不同 bitwidth 或非等价 recurrence 不会被误消除。
- [x] 3.3 运行最小相关 lit 验证，至少覆盖新增 phase2 用例与现有 `test/phase2/a5vm_fusion_region_lifecycle.mlir`，确认 pipeline 顺序和 IR 形态符合 spec。
