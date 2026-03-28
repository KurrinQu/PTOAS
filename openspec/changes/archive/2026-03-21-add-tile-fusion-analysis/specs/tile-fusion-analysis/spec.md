# Tile Fusion Analysis Specification

## ADDED Requirements

### Requirement: PreFusionAnalysisPass MUST operate in tile_buf world

`PreFusionAnalysisPass` MUST 以 `tile_buf world` IR 为输入，不依赖后段 memref-world 或 5.5 之后的 materialization 流水线。

#### Scenario: Analysis runs before memref-world lowering

- **WHEN** tile fusion planning 被启用
- **THEN** `PreFusionAnalysisPass` MUST 在 `PTOViewToMemref` 之前定义其正确性
- **AND** MUST NOT 依赖 5.5 之后的 inline / low-level fusion / load-store elimination 才能输出分析结果

### Requirement: PreFusionAnalysisPass MUST normalize SSA and DPS tile ops

`PreFusionAnalysisPass` MUST 尽量同时支持 SSA 与 DPS 两种 tile op 输入形态，并输出统一的 producer / consumer 语义。

#### Scenario: SSA and DPS inputs share one analysis model

- **WHEN** 一个 block-local 候选区域中同时出现 SSA 形式与 DPS 形式的目标 op
- **THEN** 分析层 MUST 将它们规约到统一的 tile 输入、tile 输出和依赖边模型
- **AND** MUST 为后续 planning 阶段输出一致的 DFG 与生命周期信息

### Requirement: PreFusionAnalysisPass MUST output reusable dependency metadata

分析结果 MUST 至少包含后续 planning / scheduling 可直接消费的元数据。

#### Scenario: Analysis result exposes DFG, liveness, and iteration-domain classes

- **WHEN** `PreFusionAnalysisPass` 完成
- **THEN** 其分析结果 MUST 包含：
  - block-local DFG
  - tile 生命周期区间
  - 迭代域等价类
  - hard boundary / local boundary 分类

### Requirement: PreFusionAnalysisPass MUST stay analysis-only in v1 scope

`PreFusionAnalysisPass` 在本 change 中 MUST 只产出分析结论，不承担分组或调度职责。

#### Scenario: Analysis does not emit grouping or scheduling decisions

- **WHEN** `PreFusionAnalysisPass` 在 5.1 范围内运行
- **THEN** 它 MUST NOT 直接生成 `fusion_id`、`pto.fusion.order` 或等价的分组结果
- **AND** MUST NOT 对 op 做物理重排或调度决策
- **AND** MUST 只为后续 5.3 planning / 5.4 scheduling 提供可复用的分析元数据

### Requirement: treshape MUST be treated as a local non-through boundary

`pto.treshape` 在 v1 planning 范围内 MUST 被视为局部非穿透边界，而不是可融合 compute op 或全局硬屏障。

#### Scenario: Dependency chain does not fuse through treshape

- **WHEN** 存在 `OPA -> pto.treshape -> OPB` 的依赖链
- **THEN** `PreFusionAnalysisPass` MUST NOT 为 `OPA` 与 `OPB` 建立可穿透的融合依赖关系

#### Scenario: Unrelated ops are not globally blocked by treshape

- **WHEN** 某个 op 与给定 `pto.treshape` 没有数据依赖关系
- **THEN** 分析层 MUST NOT 仅因为该 `pto.treshape` 存在，就把该 op 从同一 block-local planning 候选区域整体排除

### Requirement: Dynamic iteration-domain equality MUST stay conservative without ShapeInferencePass

当两个 op 的迭代域一致性需要依赖 5.2 的动态 shape 证明时，`PreFusionAnalysisPass` MUST 输出不可证结果。

#### Scenario: Unknown dynamic-domain equality is marked unproven

- **WHEN** 两个目标 op 的 `v_row` / `v_col` 关系无法在 5.1 阶段静态证明相等
- **THEN** 分析层 MUST 将该关系标记为不可证
- **AND** 后续 planning 阶段 MUST 可以据此保守拒绝融合
