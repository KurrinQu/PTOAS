# 任务列表: 补全缺失算子的 OpLib 支持 (trem, trems, tprelu, tlrelu)

## 阶段 1: OpLib 模板准备

- [x] 在 `oplib/level3/` 下确定或创建包含 `trem` 实现的模板文件。
- [x] 在 `oplib/level3/` 下确定或创建包含 `trems` 实现的模板文件。
- [x] 在 `oplib/level3/` 下确定或创建包含 `tprelu` 实现的模板文件。
- [x] 在 `oplib/level3/` 下确定或创建包含 `tlrelu` 实现的模板文件。

## 阶段 2: 降低逻辑实现 (Lowering)

- [x] 在 `lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp` 中实现 `TRemOp` 到 OpLib 调用的转换模式。
- [x] 在 `lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp` 中实现 `TRemSOp` 到 OpLib 调用的转换模式。
- [x] 在 `lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp` 中实现 `TPReluOp` 到 OpLib 调用的转换模式。
- [x] 在 `lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp` 中实现 `TLReluOp` 到 OpLib 调用的转换模式。

## 阶段 3: 测试与验证

- [x] 创建 `test/oplib/missing_ops_oplib.mlir` 测试文件，涵盖上述算子。
- [x] 验证 A5 架构下算子能够正确匹配模板并完成实例化 (`-pto-instantiate-and-inline-op-lib`)。
- [x] 验证最终生成的 C++ 代码符合预期且包含正确的算子实现。

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
