# Tile Fusion Scheduling Specification

## ADDED Requirements

### Requirement: OpSchedulingPass MUST stay within 5.4 scheduling scope

`OpSchedulingPass` MUST 只消费既有的 `pto.fusion.group_id` / `pto.fusion.order` 元数据做物理重排，不得重新决定 fusion group 的成员集合或组内逻辑顺序。

#### Scenario: Scheduler consumes existing planning metadata without regrouping

- **WHEN** `FusionPlanPass` 已经为一组 op 产出 `pto.fusion.group_id` / `pto.fusion.order`
- **THEN** `OpSchedulingPass` MUST 以这些 metadata 作为唯一的 group 身份与组内顺序来源
- **AND** MUST NOT 在 5.4 阶段新增、删除、拆分、合并 fusion group
- **AND** MUST NOT 重新解释哪些 op 应属于同一 fusion group

### Requirement: OpSchedulingPass MUST compact group members into contiguous block-local spans

`OpSchedulingPass` MUST 在 basic block 内将同一融合组的成员压缩成连续运行片段。

#### Scenario: Group members become contiguous in one block

- **WHEN** 一组目标 op 已拥有相同的 `pto.fusion.group_id`
- **THEN** `OpSchedulingPass` MUST 在所属 basic block 内将它们重排为连续片段
- **AND** 组内物理顺序 MUST 与 `pto.fusion.order` 一致

### Requirement: OpSchedulingPass MUST preserve legality boundaries

`OpSchedulingPass` MUST 保持 SSA、side-effect、barrier 以及 region / block 合法性，不得通过调度破坏这些边界。

#### Scenario: Scheduler does not cross hard boundaries

- **WHEN** 某次移动会跨越 barrier、side-effect op、外部 call，或跨出原有 region / block
- **THEN** `OpSchedulingPass` MUST 禁止该移动

#### Scenario: Scheduler does not move an op before its SSA definition

- **WHEN** 某次移动会让 op 出现在其某个 SSA operand 定义之前
- **THEN** `OpSchedulingPass` MUST 禁止该移动

### Requirement: Unrelated treshape MUST NOT act as a global scheduling barrier

`OpSchedulingPass` MUST NOT 将与目标 group 无依赖关系的 `pto.treshape` 当作全局调度屏障；`pto.treshape` 不属于 fusion group。

#### Scenario: Group can move across an unrelated treshape

- **WHEN** 一个 group 成员与给定 `pto.treshape` 没有数据依赖关系，且移动不会违反其它合法性约束
- **THEN** `OpSchedulingPass` MAY 让该 group 成员跨过该 `pto.treshape` 进行聚拢

#### Scenario: treshape still blocks moves that would violate SSA legality

- **WHEN** 某次跨 `pto.treshape` 的移动会破坏 SSA 定义顺序或其它合法性边界
- **THEN** `OpSchedulingPass` MUST 禁止该移动
