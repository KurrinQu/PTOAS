# Tile Fusion Region Encapsulation Specification

## ADDED Requirements

### Requirement: `PTOFusionRegionGenPass` MUST precede A5 backend lowering

当 A5 fusion 主线进入 backend lowering 时，`PTOFusionRegionGenPass` MUST 已经先把已调度的 span 封装成 `pto.fusion_region`，并把该结构化边界交给后续 A5 backend lowering 消费。

#### Scenario: Fusion region is materialized before `PTOToA5VM`

- **WHEN** A5 fusion pipeline 需要把已调度的 fusion group 继续送入 A5 backend lowering
- **THEN** `PTOFusionRegionGenPass` MUST 在 `PTOToA5VM` 之前运行
- **AND** `PTOToA5VM` MUST 以 `pto.fusion_region` 作为进入 backend lowering 前的稳定结构化边界
- **AND** backend lowering MUST NOT 跳过该封装阶段，直接回到未封装的 scheduled PTO op 序列
