# OpLib Lowering 规范: 补全缺失算子 (trem, trems, tprelu, tlrelu)

## ADDED Requirements

### Requirement: 基础算术和激活算子的 OpLib 映射支持

PTO OpLib 降低（Lowering）流水线 MUST 支持将 `trem`、`trems`、`tprelu` 和 `tlrelu` 算子映射到其对应的 OpLib 模板。

#### Scenario: 将 PTO 算子降低为 OpLib 调用

- `TRemOp` (trem) 应当被降低为使用 `trem` 模板的 OpLib 调用。
- `TRemSOp` (trems) 应当被降低为使用 `trems` 模板的 OpLib 调用。
- `TPReluOp` (tprelu) 应当被降低为使用 `tprelu` 模板的 OpLib 调用。
- `TLReluOp` (tlrelu) 应当被降低为使用 `tlrelu` 模板的 OpLib 调用，并将 `slope` 属性正确传递给模板。
- 所有降低后的调用必须根据原始 PTO 算子的具体 Tile 形状（Shape）和元素类型（Type）进行正确的实例化。
- 降低过程应当在开启 A5 架构支持且使用 OpLib 流水线时自动执行。
