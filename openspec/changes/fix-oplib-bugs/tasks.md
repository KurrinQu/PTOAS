# 任务列表: OpLib Bug 修复

## 阶段 1: 地址空间保留 (Address Space Preservation)

- [ ] 修改 `lib/PTO/Transforms/PTOInstantiateAndInlineOpLib.cpp` 中的克隆逻辑。
- [ ] 确保在实例化模板时，`AllocTileOp` 的 `addr` 属性被正确传播。

## 阶段 2: 步长推导增强 (Stride Inference)

- [ ] 修改 `lib/PTO/Transforms/InferPTOLayout.cpp` 以支持更复杂的 View 转换。
- [ ] 实现针对 `memref.subview` 衍生操作的步长计算逻辑。

## 阶段 3: 类型显式化 (Type Explicitization)

- [ ] 检查 `lib/PTO/Transforms/PTOToEmitC.cpp` 中处理混合类型运算的代码。
- [ ] 在必要的地方插入 EmitC 风格的强制类型转换。

## 阶段 4: 验证与回归

- [ ] 针对地址丢失问题编写专门的 lit 测试。
- [ ] 验证多维 View 场景下的 Layout 推导正确性。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
