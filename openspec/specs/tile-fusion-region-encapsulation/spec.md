# tile-fusion-region-encapsulation Specification

## Purpose
TBD - created by archiving change add-tile-fusion-region-encapsulation. Update Purpose after archive.
## Requirements
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

### Requirement: pto.fusion_region MUST stay as a lightweight local region container

`pto.fusion_region` MUST 作为当前函数中的最小单块容器存在，不要求把区域外依赖重写成显式闭包输入。

#### Scenario: External values remain implicit captures

- **WHEN** 融合片段中的某个 op 使用了片段外定义的 SSA value
- **THEN** `PTOFusionRegionGenPass` MAY 直接保留该 use 作为 region body 对父作用域 SSA 的隐式引用
- **AND** MUST NOT 仅为了 5.5 封装而额外生成 `pto.fusion_region` operands 和 body block arguments

#### Scenario: Escaping values become explicit region outputs

- **WHEN** 融合片段内定义的某个 SSA value 在片段外仍然被使用
- **THEN** 该 value MUST 作为 `pto.yield` operand 显式返回
- **AND** `pto.fusion_region` MUST 产生与之对应的 result 供区域外继续使用

#### Scenario: Yield set marks the region-external tile frontier

- **WHEN** `PTOFusionRegionGenPass` 为一个 `pto.fusion_region` 构造 `pto.yield`
- **THEN** `pto.yield` operands MUST 恰好枚举封装后仍对 region 外可见的 tile/value 集合
- **AND** 未出现在 `pto.yield` 中的中间 tile MUST 视为 region-local
- **AND** 后续 region-local 冗余 load/store 消除 MAY 直接把该 `pto.yield` 集合作为外部可见性边界

#### Scenario: Dead terminal values do not become region outputs

- **WHEN** 融合片段末尾写入的 destination tile 在片段外没有任何 SSA use
- **THEN** `PTOFusionRegionGenPass` MUST NOT 仅因为它是“最后一个产出”就为其生成 `pto.yield` operand
- **AND** 对应 `pto.fusion_region` MAY 保持空 result 列表

#### Scenario: Region body keeps no explicit entry arguments

- **WHEN** `PTOFusionRegionGenPass` 完成一个 `pto.fusion_region` 的构造
- **THEN** region body MUST 保持单 block 且 entry block arguments 为空
- **AND** 外部依赖 MUST 通过父作用域 SSA 隐式捕获而不是 body block arguments 传入

### Requirement: pto.yield MUST define the stable external result order

`pto.yield` MUST 作为 `pto.fusion_region` 的唯一 terminator，并定义该 region 对外结果的稳定顺序。

#### Scenario: Yield order matches fusion_region result order

- **WHEN** 一个 `pto.fusion_region` 需要向区域外返回多个结果
- **THEN** `pto.yield` operands 的顺序 MUST 与 `pto.fusion_region` results 的顺序一一对应
- **AND** 该顺序 MUST 稳定且可重复，不得依赖非确定性遍历顺序

### Requirement: pto.yield frontier MUST preserve stable yielded order without auxiliary frontier-class metadata

`PTOFusionRegionGenPass` MUST 仅通过显式 yielded value 列表表达 region 的对外可见 frontier。该 yielded frontier MUST 保持稳定顺序，且实现 MUST NOT 再并行生成额外的 frontier-class metadata。

#### Scenario: Yielded frontier follows stable span order

- **WHEN** 一个 `pto.fusion_region` 需要为多个仍对 region 外可见的 tile 生成 result / yield
- **THEN** `PTOFusionRegionGenPass` MUST 按 fused span 中稳定的枚举顺序生成 `pto.yield` operand 与 `pto.fusion_region` result
- **AND** 下游 store-elision MUST 仅通过该显式 yielded frontier 判断“哪些值仍对 region 外可见”

#### Scenario: Internal-dead values never receive frontier slots

- **WHEN** 某个 region 内定义的 tile 在封装后不再对 region 外可见
- **THEN** 它 MUST NOT 出现在 `pto.yield` 中
- **AND** MUST NOT 在 yielded frontier 中占位或生成伪 result

### Requirement: Illegal or incomplete scheduled input MUST be rejected explicitly

若输入不满足 5.5 封装前置条件，`PTOFusionRegionGenPass` MUST 显式失败，而不是在本阶段做隐式补救。

#### Scenario: Split spans for one group are rejected

- **WHEN** 同一 `pto.fusion.group_id` 在同一个 basic block 中出现多个非连续片段
- **THEN** `PTOFusionRegionGenPass` MUST 报错失败

#### Scenario: Incomplete fusion metadata is rejected

- **WHEN** 某个候选组成员只带有 `pto.fusion.group_id` 或只带有 `pto.fusion.order`，而不是同时具备两者
- **THEN** `PTOFusionRegionGenPass` MUST 报错失败

### Requirement: `PTOFusionRegionGenPass` MUST precede A5 backend lowering

当 A5 fusion 主线进入 backend lowering 时，`PTOFusionRegionGenPass` MUST 已经先把已调度的 span 封装成 `pto.fusion_region`，并把该结构化边界交给后续 A5 backend lowering 消费。

#### Scenario: Fusion region is materialized before `PTOToA5VM`

- **WHEN** A5 fusion pipeline 需要把已调度的 fusion group 继续送入 A5 backend lowering
- **THEN** `PTOFusionRegionGenPass` MUST 在 `PTOToA5VM` 之前运行
- **AND** `PTOToA5VM` MUST 以 `pto.fusion_region` 作为进入 backend lowering 前的稳定结构化边界
- **AND** backend lowering MUST NOT 跳过该封装阶段，直接回到未封装的 scheduled PTO op 序列
