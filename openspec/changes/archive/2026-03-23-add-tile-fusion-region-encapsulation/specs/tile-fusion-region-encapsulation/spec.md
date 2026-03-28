# Tile Fusion Region Encapsulation Specification

## ADDED Requirements

### Requirement: PTOFusionRegionGenPass MUST stay within 5.5 encapsulation scope

`PTOFusionRegionGenPass` MUST 只消费既有的 `pto.fusion.group_id` / `pto.fusion.order` 和 5.4 调度后的连续片段，负责结构化封装，不得重新决定 group 成员集合或物理顺序。

#### Scenario: Encapsulation consumes scheduled spans without regrouping or reordering

- **WHEN** `OpSchedulingPass` 已经把同一 `pto.fusion.group_id` 的成员压缩为连续片段
- **THEN** `PTOFusionRegionGenPass` MUST 直接消费该连续片段生成 `pto.fusion_region`
- **AND** MUST NOT 新增、删除、拆分或合并 fusion group
- **AND** MUST NOT 在 5.5 阶段重新调度组内成员顺序

### Requirement: 5.5 output MUST be a PTO fusion region rather than a helper function

tile fusion 5.5 的正式输出 MUST 是 PTO dialect 内的结构化 region 容器，而不是 `@__pto_fused_group_*` helper function。

#### Scenario: One scheduled group becomes one pto.fusion_region

- **WHEN** 一个 fusion group 在单个 basic block 中已经形成合法连续片段
- **THEN** `PTOFusionRegionGenPass` MUST 用一个且仅一个 `pto.fusion_region` 替换该片段
- **AND** MUST NOT 把该片段正式封装成 `@__pto_fused_group_*` helper function 作为 5.5 输出

### Requirement: pto.fusion_region MUST expose explicit region inputs and outputs

`pto.fusion_region` MUST 通过显式 operands、block arguments 和 `pto.yield` operands 表达融合区域的外部输入与外部可见输出。

#### Scenario: External values become explicit region inputs

- **WHEN** 融合片段中的某个 op 使用了片段外定义的 SSA value
- **THEN** 该 value MUST 作为 `pto.fusion_region` 的显式 input 暴露
- **AND** 对应 value MUST 通过 region body block argument 进入内部使用

#### Scenario: Escaping values become explicit region outputs

- **WHEN** 融合片段内定义的某个 SSA value 在片段外仍然被使用
- **THEN** 该 value MUST 作为 `pto.yield` operand 显式返回
- **AND** `pto.fusion_region` MUST 产生与之对应的 result 供区域外继续使用

#### Scenario: Dead terminal values do not become region outputs

- **WHEN** 融合片段末尾写入的 destination tile 在片段外没有任何 SSA use
- **THEN** `PTOFusionRegionGenPass` MUST NOT 仅因为它是“最后一个产出”就为其生成 `pto.yield` operand
- **AND** 对应 `pto.fusion_region` MAY 保持空 result 列表

### Requirement: pto.fusion_region MUST be structurally closed

封装完成后的 `pto.fusion_region` MUST 构成闭包区域，内部不得保留未通过显式输入引入的外部 SSA capture。

#### Scenario: Region body has no implicit external capture

- **WHEN** `PTOFusionRegionGenPass` 完成一个 `pto.fusion_region` 的构造
- **THEN** region body 内的所有外部依赖 MUST 只通过 block arguments 表达
- **AND** body 内 MUST NOT 直接引用未声明的区域外 SSA value

### Requirement: pto.yield MUST define the stable external result order

`pto.yield` MUST 作为 `pto.fusion_region` 的唯一 terminator，并定义该 region 对外结果的稳定顺序。

#### Scenario: Yield order matches fusion_region result order

- **WHEN** 一个 `pto.fusion_region` 需要向区域外返回多个结果
- **THEN** `pto.yield` operands 的顺序 MUST 与 `pto.fusion_region` results 的顺序一一对应
- **AND** 该顺序 MUST 稳定且可重复，不得依赖非确定性遍历顺序

### Requirement: Illegal or incomplete scheduled input MUST be rejected explicitly

若输入不满足 5.5 封装前置条件，`PTOFusionRegionGenPass` MUST 显式失败，而不是在本阶段做隐式补救。

#### Scenario: Split spans for one group are rejected

- **WHEN** 同一 `pto.fusion.group_id` 在同一个 basic block 中出现多个非连续片段
- **THEN** `PTOFusionRegionGenPass` MUST 报错失败

#### Scenario: Incomplete fusion metadata is rejected

- **WHEN** 某个候选组成员只带有 `pto.fusion.group_id` 或只带有 `pto.fusion.order`，而不是同时具备两者
- **THEN** `PTOFusionRegionGenPass` MUST 报错失败
