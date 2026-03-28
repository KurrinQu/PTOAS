# Proposal: 在 A5 fusion 主线中消除冗余 `plt` predicate 物化

## 概述

当前 A5 fusion 主线在 `PTOLowLevelLoopFusion` 后已经接入一轮通用 `CSE`，但融合后区域内仍会残留多个语义等价的 `a5vm.plt_b{8,16,32}`。这些重复 `plt` 继续携带等价的 `mask` / `scalar_out` 链，放大 region 内部 IR 噪声，也让后续 cleanup 与调试更难聚焦真正的 data path。

本 change 为 A5 fusion 主线补一个专门的 fusion-region-local predicate-elision 阶段：在 `pto.fusion_region` 仍然显式存在时，识别并复用等价 `plt` 的双结果，删除冗余 predicate 物化。

## 背景与动机

当前 `tools/ptoas/ptoas.cpp` 的 A5 fusion 主线顺序是：

`PTOToA5VM -> PTOLowLevelLoopFusion -> CSE -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion`

实际 `build/run.log` 已显示，即使在该轮 `CSE` 之后，fusion region 内仍会保留多组 `a5vm.plt_b32`。这些重复 op 并不总是共享同一个 scalar SSA，而是经常出现在 low-level loop 的 `iter_args` 链上：两个 loop-carried scalar 虽然是不同 block argument，但它们由相同初始化值出发，并持续由等价 `plt.scalar_out` 递推，最终产生相同的 `mask` / `scalar_out`。

如果继续只依赖通用 `CSE`，这类“值相同但 SSA 不同”的 predicate 物化会长期滞留在 fusion region 内，导致：

1. region 内低层 A5VM IR 可读性下降，真实计算链被重复 `plt` 干扰；
2. 后续 cleanup pass 无法直接消费更精简的 predicate 结果；
3. 对 `plt_b8` / `plt_b16` / `plt_b32` family 的行为缺少统一、可验证的 region-local 契约。

## 目标

- 在 A5 fusion 主线中新增一个专门的 `plt` 冗余消除阶段，固定放在当前内部 `CSE` 之后、`PTOFusionLoadStoreElision` 之前。
- 仅处理 `pto.fusion_region` 内部的 A5VM IR，不扩张到非 fusion 的普通 A5VM block。
- 统一覆盖 `a5vm.plt_b8`、`a5vm.plt_b16`、`a5vm.plt_b32`。
- 当两个 `plt` 的输出值等价时，复用前一个支配 `plt` 的 `mask` 与 `scalar_out`，并删除后一个冗余 op。
- 显式覆盖 low-level loop `iter_args` 的 loop-carried 等价传播，使当前日志中的 `%arg11/%arg12` 类模式可被识别。

## 非目标

- 不把该 pass 扩展为全函数范围的通用 A5VM CSE。
- 不改变 `PTOFlattenFusionRegionPass` 之后的全局 `CSE` 语义。
- 不修改 `PTOFusionLoadStoreElisionPass` 的 store frontier 契约，也不把 predicate 复用与 store 删除混成一个 capability。
- 不引入新的 CLI 选项、debug 开关或 backend 切换行为。
- 不尝试对任意复杂控制流做全局值等价证明；无法证明时保持保守不变。

## 能力变更

### New Capabilities

- `tile-fusion-predicate-elision`：定义 fusion region 内 `a5vm.plt_b{8,16,32}` 的等价判定、loop-carried scalar 等价传播，以及双结果复用/删除冗余 op 的契约。

### Modified Capabilities

- `a5vm-backend-pipeline`：补充 A5 fusion 主线在 `CSE` 与 `PTOFusionLoadStoreElision` 之间新增 predicate-elision 阶段的顺序约束。

## 影响

- Affected code:
  - `tools/ptoas/ptoas.cpp`
  - `include/PTO/Transforms/Passes.h`
  - `include/PTO/Transforms/Passes.td`
  - `lib/PTO/Transforms/TileFusion/`
  - `test/phase2/`
- Affected systems:
  - A5 fusion backend mainline
  - fusion-region-local cleanup contract
  - A5VM IR regression surface around `plt` family
- Public API / CLI:
  - 无新增用户可见 CLI；新增内部 pass entrypoint 与 pass registration

## 预期结果

- A5 fusion 主线在 flatten 之前即可消除 fusion region 内等价的 `plt` 物化。
- `a5vm.plt_b8` / `plt_b16` / `plt_b32` 在 region-local cleanup 上拥有统一契约，而不是只依赖通用 `CSE` 的偶然命中。
- 对 loop-carried scalar 递推形成的重复 `plt`，IR 能稳定收敛到单一 predicate 结果源。

## 成功标准

- OpenSpec 中新增 `tile-fusion-predicate-elision` capability，覆盖顺序、作用域、family 覆盖范围、loop-carried 等价与保守边界。
- OpenSpec 中对 `a5vm-backend-pipeline` 的增量明确新的 pass 顺序：`PTOLowLevelLoopFusion -> CSE -> predicate-elision -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion`。
- 变更后测试可验证：
  - fusion region 内重复 `plt_b32` 被折叠为单一结果源；
  - `plt_b8` / `plt_b16` / `plt_b32` family 都受同一契约约束；
  - 非等价 loop-carried scalar、不同 bitwidth 或无法证明等价的场景保持不变。
