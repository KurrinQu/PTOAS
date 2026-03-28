# Tile Fusion Region Lowering Specification

## ADDED Requirements

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

### Requirement: `pto.fusion_region` MUST remain the structured lowering boundary until explicit flatten

在 5.5 之后到 Emit/手工降级之前，tile fusion 主线 MUST 持续以 `pto.fusion_region` 作为结构化变换边界，直到显式 flatten 步骤运行。

#### Scenario: Downstream tile-fusion passes transform region body in place

- **WHEN** `PTOInstantiateAndLowerToLibCall`、`PTOInlineLibCall` 或 `PTOLowLevelLoopFusion` 处理一个已经封装成 `pto.fusion_region` 的 fusion group
- **THEN** 这些 pass MUST 在原函数中的 region body 内原位变换该 group
- **AND** MUST NOT 将该 group 重新 outline 成 `@__pto_fused_group_*` helper function 作为 tile fusion 主线的正式中间态

#### Scenario: Region frontier remains explicit until flatten

- **WHEN** 下游 pass 在 `pto.fusion_region` 内完成 lowering、inline 或 low-level fusion
- **THEN** `pto.yield` 与 `pto.fusion_region` results MUST 继续作为该 region 对外可见值的唯一正式边界
- **AND** 下游 pass MUST NOT 隐式绕过 `pto.yield` / region result frontier 直接改写父 block 中的 escaping SSA uses

### Requirement: `PTOFlattenFusionRegionPass` MUST eliminate residual fusion regions before Emit

`PTOFlattenFusionRegionPass` MUST 作为 `pto.fusion_region` 的正式消解出口，并确保 Emit/手工降级之前不再残留该 wrapper。

#### Scenario: Flatten splices region body back to parent block

- **WHEN** `PTOFlattenFusionRegionPass` 处理一个带 `pto.yield` 的 `pto.fusion_region`
- **THEN** 它 MUST 将 region body 中除 `pto.yield` 之外的 op splice 回父 block
- **AND** MUST 用 `pto.yield` operands 替换该 region 的 results
- **AND** MUST 删除 `pto.yield` 与 `pto.fusion_region`

#### Scenario: Empty-yield region flattens without synthetic outputs

- **WHEN** `PTOFlattenFusionRegionPass` 处理一个 result 列表为空、并以空 `pto.yield` 结束的 `pto.fusion_region`
- **THEN** 它 MUST 直接将 region body splice 回父 block 并删除 wrapper
- **AND** MUST NOT 仅为了 flatten 构造额外 placeholder result 或 synthetic store

#### Scenario: No residual fusion_region reaches Emit pipeline

- **WHEN** tile fusion 主线进入 Emit/手工降级相关 pass
- **THEN** IR 中 MUST NOT 再残留 `pto.fusion_region` 或 `pto.yield`

