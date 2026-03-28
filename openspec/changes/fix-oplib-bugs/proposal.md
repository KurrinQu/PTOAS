# Proposal: 修复 OpLib 流程中的地址空间与类型推导缺陷

## 目标 (Goals)

解决 A5 架构下 OpLib 实例化过程中出现的硬件相关属性丢失及类型不匹配问题。

## 现状 (Current State)

在 `level3` 编译模式下，`AllocTileOp` 必须携带明确的 `addr` (地址空间)。但在 OpLib 模板内联后，部分地址信息未能正确传递。此外，Tile 的 Stride (步长) 在经过 View 变换后可能推导错误。

## 预期行为 (Expected Behavior)

1. 确保 `AllocTileOp` 的地址属性在变换过程中得以保留。
2. 修复 Layout 推导中关于 SubView/Reshape 的步长计算逻辑。
3. 在 EmitC codegen 阶段增加必要的类型转换。

## 影响范围 (Scope)

- `lib/PTO/Transforms/PTOInstantiateAndInlineOpLib.cpp`
- `lib/PTO/Transforms/InferPTOLayout.cpp`
- `lib/PTO/Transforms/PTOToEmitC.cpp`
