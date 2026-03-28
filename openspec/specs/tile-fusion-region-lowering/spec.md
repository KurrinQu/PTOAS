# tile-fusion-region-lowering Specification

## Purpose
TBD - created by archiving change advance-tile-fusion-lower-to-libcall. Update Purpose after archive.
## Requirements
### Requirement: `pto.fusion_region` MUST be transparent to pre-lowering memory and sync passes

在当前 memref-world 过渡主线中，`PlanMemory` 与 `PTOInsertSync` MUST 把 `pto.fusion_region` 视为局部结构化容器，而不是未知 compute wrapper。

#### Scenario: PlanMemory analyzes local buffers inside fusion_region

- **WHEN** `PlanMemory` 处理一个已经过 `PTOViewToMemref`、且仍包含 `pto.fusion_region` 的函数
- **THEN** 它 MUST 递归分析 region body 内的 local buffer 读写与生命周期，而不是把 `pto.fusion_region` 当作“touches local buffer”的未知 op
- **AND** 它 MUST 建立 `pto.yield` operands 与 `pto.fusion_region` results 之间的 alias / external-frontier 关系
- **AND** 它 MUST NOT 仅因为 local buffer 位于 region body 内部就失败

#### Scenario: InsertSync preserves region-internal dependencies without wrapper sync

- **WHEN** `PTOInsertSync` 处理一个包含 `pto.fusion_region` 的函数
- **THEN** 它 MUST 递归进入 region body，并基于其中的实际 op 构建 dependency 与 sync 分析
- **AND** 它 MUST 把 `pto.yield` 视为 region 对外可见结果的 frontier
- **AND** 它 MUST NOT 仅为了 `pto.fusion_region` wrapper 本身额外生成独立 sync boundary

### Requirement: `pto.fusion_region` MUST remain the structured lowering boundary for planned fusion groups until explicit flatten

在 5.5 之后到 A5VM emission / EmitC 手工降级之前，tile fusion 主线 MUST 对已经计划并封装的 fusion group 持续以 `pto.fusion_region` 作为结构化变换边界，直到显式 flatten 步骤运行。
在该边界保持不变的前提下，`PTOA5VMVersionSelectionPass` MAY 为 `pto.fusion_region` 内部 PTO op 写入 `pto.a5vm_lowering_choice`，`PTOToA5VM` 与后续 pass 也 MAY 消费这些 per-op 属性；这类属性 MUST NOT 改变 `pto.yield`、region result 或父 block SSA use 的正式 frontier。

#### Scenario: Downstream tile-fusion passes transform region body in place

- **WHEN** `PTOA5VMVersionSelectionPass`、`PTOToA5VM` 或 `PTOLowLevelLoopFusion` 处理一个已经封装成 `pto.fusion_region` 的 fusion group
- **THEN** 这些 pass MUST 在原函数中的 region body 内原位变换该 group
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
