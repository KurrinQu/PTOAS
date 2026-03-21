# Tile Fusion Planning Specification

## ADDED Requirements

### Requirement: FusionPlanPass MUST stay within 5.3 planning scope

`FusionPlanPass` MUST 只在 5.3 planning 阶段内做“选哪些 op 成组、以及组内逻辑顺序如何表达”的决策，不承担 5.4 scheduling 的物理聚拢职责。

#### Scenario: Planning emits metadata but does not reorder ops

- **WHEN** `FusionPlanPass` 在 tile fusion pipeline 中运行
- **THEN** 它 MUST 只输出 block-local DAG 分组结论和 `pto.fusion.*` metadata
- **AND** MUST NOT 改变 op 的物理顺序
- **AND** MUST NOT 把 basic block 内的离散 group member 聚拢成连续片段
- **AND** 后续物理重排职责 MUST 留给独立的 `OpSchedulingPass`

### Requirement: FusionPlanPass MUST build fusion groups from block-local DAGs

`FusionPlanPass` MUST 基于 `PreFusionAnalysisPass` 提供的 block-local DFG 进行分组，而不是仅基于线性连续链。

#### Scenario: Diamond subgraph can form one fusion group

- **WHEN** 同一 basic block 内存在 `tmax -> tsub x2 -> texp x2 -> tmul x2 -> tadd` 这类无环、同迭代域的子图
- **THEN** `FusionPlanPass` MUST 允许将其识别为一个合法融合组

#### Scenario: Join subgraph can form one fusion group

- **WHEN** 同一 basic block 内存在两路 `trowexpandmul` 汇合到 `tadd` 的 join 子图，且 legality 检查通过
- **THEN** `FusionPlanPass` MUST 允许将其识别为一个合法融合组

### Requirement: FusionPlanPass MUST cover the driver-sample op closure

v1 planning 范围 MUST 覆盖当前 driver sample 所需的最小 op 闭包。

#### Scenario: Planning accepts current driver-sample compute families

- **WHEN** 候选 op 属于以下集合：
  - 当前已支持的 binary / binary-scalar elementwise
  - `texp`
  - `texpands`
  - `trowexpandmul`
  - `trowexpanddiv`
- **THEN** `FusionPlanPass` MUST 将其视为可规划 compute family

### Requirement: treshape MUST only block the dependency chain that passes through it

`FusionPlanPass` MUST 吃进 `treshape` 的局部非穿透语义，而不是把它当作全局 planning 屏障。

#### Scenario: Fusion does not pass through treshape on the same dependency chain

- **WHEN** 存在 `OPA -> pto.treshape -> OPB` 的依赖链
- **THEN** `FusionPlanPass` MUST NOT 因该链路把 `OPA` 与 `OPB` 规划到同一融合组

#### Scenario: Unrelated candidates are not globally blocked by treshape

- **WHEN** 某个候选 op 与给定 `pto.treshape` 没有数据依赖关系
- **THEN** `FusionPlanPass` MUST NOT 仅因为该 `pto.treshape` 位于同一 block 中，就拒绝该候选 op 与其他合法 op 成组

### Requirement: FusionPlanPass MUST emit stable group metadata

`FusionPlanPass` MUST 为每个被选中的融合组输出稳定、可调度的元数据。

#### Scenario: Planning emits fusion_id and stable in-group order

- **WHEN** 一个候选子图被接受为合法融合组
- **THEN** 组内所有成员 MUST 获得同一个 `pto.fusion.group_id`
- **AND** 每个成员 MUST 获得稳定的 `pto.fusion.order`
- **AND** `pto.fusion.order` MUST 先满足拓扑依赖，再尽量贴近原始程序顺序
- **AND** metadata MUST 足以让后续 `OpSchedulingPass` 在不重新决策组成员集合的前提下直接消费

### Requirement: Unproven dynamic iteration domains MUST be rejected conservatively

当融合 legality 依赖动态迭代域等价，而 5.2 尚未提供证明时，`FusionPlanPass` MUST 输出保守拒绝结论，而不是猜测两个 op 可成组。

#### Scenario: Dynamic shape equality not proven blocks grouping

- **WHEN** 融合 legality 依赖动态 shape 等价，而 5.2 尚未提供证明
- **THEN** `FusionPlanPass` MUST 保守拒绝该候选组
