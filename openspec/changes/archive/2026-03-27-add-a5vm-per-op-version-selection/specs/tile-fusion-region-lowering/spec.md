# Tile Fusion Region Lowering Specification

## MODIFIED Requirements

### Requirement: `pto.fusion_region` MUST remain the structured lowering boundary for planned fusion groups until explicit flatten

在 5.5 之后到 A5VM emission / EmitC 手工降级之前，tile fusion 主线 MUST 对已经计划并封装的 fusion group 持续以 `pto.fusion_region` 作为结构化变换边界，直到显式 flatten 步骤运行。  
在该边界保持不变的前提下，`PTOA5VMVersionSelectionPass` MAY 为 `pto.fusion_region` 内部 PTO op 写入 `pto.a5vm_lowering_choice`，`PTOToA5VM` 与后续 pass 也 MAY 消费这些 per-op 属性；这类属性 MUST NOT 改变 `pto.yield`、region result 或父 block SSA use 的正式 frontier。

#### Scenario: Downstream tile-fusion passes transform region body in place

- **WHEN** `PTOA5VMVersionSelectionPass`、`PTOToA5VM` 或 `PTOLowLevelLoopFusion` 处理一个已经封装成 `pto.fusion_region` 的 fusion group
- **THEN** 这些 pass MUST 在原函数中的 region body 内原位标注或变换该 group
- **AND** MUST NOT 将该 group 重新 outline 成 `@__pto_fused_group_*` helper function 作为 tile fusion 主线的正式中间态

#### Scenario: Residual non-fused PTO ops stay outside `pto.fusion_region`

- **WHEN** `PTOFusionRegionGenPass` 之后仍然存在未被 fusion planning 接纳的 PTO op
- **THEN** 这些 residual non-fused PTO op MAY 保持在 `pto.fusion_region` 外部
- **AND** `PTOA5VMVersionSelectionPass` MUST 继续为其中属于 A5VM candidate 的 op 标注 `pto.a5vm_lowering_choice`
- **AND** `PTOToA5VM` MAY 继续在原 block 中直接降低这些 residual op
- **AND** downstream pass MUST NOT 仅为了满足 backend lowering 或版本选择而把这类 residual op 人工补包成新的 `pto.fusion_region`

#### Scenario: Region frontier remains explicit until flatten

- **WHEN** 下游 pass 在 `pto.fusion_region` 内完成版本选择、A5VM lowering 或 post-lowering low-level fusion
- **THEN** `pto.yield` 与 `pto.fusion_region` results MUST 继续作为该 region 对外可见值的唯一正式边界
- **AND** 下游 pass MUST NOT 隐式绕过 `pto.yield` / region result frontier 直接改写父 block 中的 escaping SSA uses
- **AND** MUST NOT 因为新增 per-op 版本选择属性而改变 `pto.fusion_region` 的结构化边界语义
