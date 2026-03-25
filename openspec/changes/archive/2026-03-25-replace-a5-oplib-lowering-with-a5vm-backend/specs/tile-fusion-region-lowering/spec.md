# Tile Fusion Region Lowering Specification

## MODIFIED Requirements

### Requirement: `pto.fusion_region` MUST remain the structured lowering boundary for planned fusion groups until explicit flatten

在 5.5 之后到 A5VM emission / EmitC 手工降级之前，tile fusion 主线 MUST 对已经计划并封装的 fusion group 持续以 `pto.fusion_region` 作为结构化变换边界，直到显式 flatten 步骤运行。

#### Scenario: Downstream tile-fusion passes transform region body in place

- **WHEN** `PTOToA5VM` 或 `PTOLowLevelLoopFusion` 处理一个已经封装成 `pto.fusion_region` 的 fusion group
- **THEN** 这些 pass MUST 在原函数中的 region body 内原位变换该 group
- **AND** MUST NOT 将该 group 重新 outline 成 `@__pto_fused_group_*` helper function 作为 tile fusion 主线的正式中间态

#### Scenario: Residual non-fused PTO ops stay outside `pto.fusion_region`

- **WHEN** `PTOFusionRegionGenPass` 之后仍然存在未被 fusion planning 接纳的 PTO op
- **THEN** 这些 residual non-fused PTO op MAY 保持在 `pto.fusion_region` 外部
- **AND** `PTOToA5VM` MAY 继续在原 block 中直接降低这些 residual op
- **AND** downstream pass MUST NOT 仅为了满足 backend lowering 而把这类 residual op 人工补包成新的 `pto.fusion_region`

#### Scenario: Region frontier remains explicit until flatten

- **WHEN** 下游 pass 在 `pto.fusion_region` 内完成 A5VM lowering 或 post-lowering low-level fusion
- **THEN** `pto.yield` 与 `pto.fusion_region` results MUST 继续作为该 region 对外可见值的唯一正式边界
- **AND** 下游 pass MUST NOT 隐式绕过 `pto.yield` / region result frontier 直接改写父 block 中的 escaping SSA uses

### Requirement: `PTOFlattenFusionRegionPass` MUST eliminate residual planned fusion regions before Emit

`PTOFlattenFusionRegionPass` MUST 作为 planned fusion group 所对应 `pto.fusion_region` 的正式消解出口，并确保 A5VM emission / EmitC 手工降级之前不再残留该 wrapper。

#### Scenario: Flatten splices region body back to parent block

- **WHEN** `PTOFlattenFusionRegionPass` 处理一个带 `pto.yield` 的 `pto.fusion_region`
- **THEN** 它 MUST 将 region body 中除 `pto.yield` 之外的 op splice 回父 block
- **AND** MUST 用 `pto.yield` operands 替换该 region 的 results
- **AND** MUST 删除 `pto.yield` 与 `pto.fusion_region`

#### Scenario: Empty-yield region flattens without synthetic outputs

- **WHEN** `PTOFlattenFusionRegionPass` 处理一个 result 列表为空、并以空 `pto.yield` 结束的 `pto.fusion_region`
- **THEN** 它 MUST 直接将 region body splice 回父 block 并删除 wrapper
- **AND** MUST NOT 仅为了 flatten 构造额外 placeholder result 或 synthetic store

#### Scenario: No residual fusion_region reaches backend emission

- **WHEN** tile fusion 主线进入 A5VM emission 或 EmitC/手工降级相关 pass
- **THEN** IR 中 MUST NOT 再残留 `pto.fusion_region` 或 `pto.yield`
