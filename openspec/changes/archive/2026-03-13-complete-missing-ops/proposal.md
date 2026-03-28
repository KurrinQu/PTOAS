# Proposal: 补全缺失算子的 OpLib 支持 (trem, trems, tprelu, tlrelu)

## 目标 (Goals)

补全 `trem` (取模)、`trems` (标量取模)、`tprelu` (参数化 ReLU) 和 `tlrelu` (Leaky ReLU) 算子的 OpLib 映射与模板支持，确保在 A5 架构下能够正确降低到 OpLib 调用并生成高效代码。

## 现状 (Current State)

目前 `pto` 方言已经在 `PTOOps.td` 中定义了 `trem`、`trems`、`tprelu` 和 `tlrelu` 算子，并且具备基础的验证器。然而，在 A5 架构的 OpLib 降低流程（`PTOLowerToOpLibCalls.cpp`）中尚未建立这些算子到 OpLib 模板的映射，且 `oplib/` 目录下也缺失相应的 `.mlir` 模板文件。这导致在开启 OpLib 优化时，这些算子无法被正确处理。

## 预期行为 (Expected Behavior)

1. 在 `oplib/level3/` 中新增或更新模板文件，包含上述算子的实现。
2. 在 `PTOLowerToOpLibCalls.cpp` 中添加映射逻辑，将 PTO 算子连接到相应的 OpLib 模板。
3. 确保这些算子能够通过 OpLib 降低、实例化、内联以及 C++ 代码生成的完整流程。

## 影响范围 (Scope)

- `oplib/level3/*.mlir` (新增/修改模板)
- `lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp` (添加降低模式)
- `test/oplib/` (新增端到端测试用例)
