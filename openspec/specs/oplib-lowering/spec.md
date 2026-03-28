# OpLib Lowering Specification

## Purpose

定义 PTO IR 到 OP-Lib 调用的 lowering 契约，覆盖模板匹配、实例化与强制门禁行为，确保新增算子和既有算子都能以可维护、可验证的方式接入 OP-Lib 流水线。
## Requirements
### Requirement: 基础算术和激活算子的 OpLib 映射支持

PTO OpLib 降低（Lowering）流水线 MUST 支持将 `trem`、`trems`、`tprelu` 和 `tlrelu` 算子映射到其对应的 OpLib 模板。

#### Scenario: 将 PTO 算子降低为 OpLib 调用

- `TRemOp` (`trem`) 应当被降低为使用 `trem` 模板的 OpLib 调用。
- `TRemSOp` (`trems`) 应当被降低为使用 `trems` 模板的 OpLib 调用。
- `TPReluOp` (`tprelu`) 应当被降低为使用 `tprelu` 模板的 OpLib 调用。
- `TLReluOp` (`tlrelu`) 应当被降低为使用 `tlrelu` 模板的 OpLib 调用，并将 `slope` 属性正确传递给模板。
- 所有降低后的调用必须根据原始 PTO 算子的具体 Tile 形状（Shape）和元素类型（Type）进行正确的实例化。
- 降低过程应当在开启 A5 架构支持且使用 OpLib 流水线时自动执行。

### Requirement: Interface-based OpLib lowering

The OpLib lowering mechanism SHALL be refactored to use an interface-driven approach to improve extensibility and maintainability.

#### Scenario: Replace hardcoded matching with interfaces

- A new `OpLibOpInterface` SHALL be defined to provide OpLib template information.
- The `PTOLowerToOpLibCalls` pass SHALL be refactored to use `OpLibOpInterface` instead of hardcoded operator matching.
- Existing PTO operators SHALL implement `OpLibOpInterface` to maintain functionality.

#### Scenario: Interface descriptor fields preserve legacy matching semantics

- `OpLibOpInterface` descriptors SHALL carry enough data to reconstruct `MatchRequest`, including `kind/opName`, operand order and roles, and optional fields `scalarPos/cmpMode/isBinary/requiredVariantId`.
- Ops that require temporary tile operands, for example `tprelu`, `trowsum`, `trowmax`, `trowmin`, and `tcolsum`, SHALL expose `tmp` in descriptor operands with correct role and order.
- Interface-driven matching SHALL preserve legacy constraints for active compare/select lowering (`tcmp`, `tcmps`, `tsel`) and MUST reject or bypass mismatched select-scalar paths such as `tsels` when the PTO IR contract does not match the available templates.
- Interface-driven matching SHALL preserve `tdivs` variant restriction via `requiredVariantId` (`tile_scalar` or `scalar_tile`).
- Interface-driven matching SHALL preserve `tcolsum` branch selection via `isBinary`.
- Interface-driven matching SHALL preserve `tprelu` tmp element type (`i8`) and `trem`/`trems` float-only restrictions.

### Requirement: 4.5~4.9 target ops must not silently fallback

For ops in PTO IR manual sections 4.5~4.9 (current lowering target set), OP-Lib lowering SHALL be mandatory once `PTOInstantiateAndLowerToLibCall` runs.

#### Scenario: Mandatory lowering for target op family

- During OP-Lib lowering, if a target op fails at descriptor-to-`MatchRequest`, candidate selection, instance creation, or call rewrite, the pass SHALL emit error and fail.
- The pass SHALL NOT keep original target ops via warning-based fallback for these families.

#### Scenario: Post-pass no-residual check

- After lowering, non-OPLib user functions SHALL NOT contain remaining target PTO ops.
- If any residual target op exists, the pass SHALL emit error and fail.

### Requirement: Grouped OpLib lowering MUST consume `pto.fusion_region` as the post-5.5 lowering unit

在 tile fusion 5.5 之后，grouped `PTOInstantiateAndLowerToLibCall` MUST 直接消费 `pto.fusion_region`，而不是继续依赖已经被封装阶段移除的 per-op `pto.fusion.*` metadata。

#### Scenario: Region body is lowered in place after encapsulation

- **WHEN** 一个 fusion group 已经由 `PTOFusionRegionGenPass` 封装成 `pto.fusion_region`
- **AND** 该 region body 内的 compute op 都属于当前 active grouped OpLib lowering scope
- **THEN** `PTOInstantiateAndLowerToLibCall` MUST 在该 region body 内原位把这些 compute op 改写为 OP-Lib call
- **AND** MUST 保留 `pto.yield` / region result frontier，直到显式 flatten 运行

#### Scenario: Grouped lowering does not require stale per-op fusion metadata

- **WHEN** `PTOFusionRegionGenPass` 已经移除组内成员上的 `pto.fusion.group_id` / `pto.fusion.order`
- **THEN** grouped `PTOInstantiateAndLowerToLibCall` MUST 仍能仅依靠 `pto.fusion_region` 本身识别该 lowering unit
- **AND** MUST NOT 要求 region body 内成员继续保留旧的 per-op `pto.fusion.*` attrs

### Requirement: Region-based grouped lowering MUST remain interface-driven and deterministic in memref world

即使 tile fusion 主线当前仍处于 memref-world `PlanMemory` / `InsertSync` 之后，region-based grouped lowering 也 MUST 继续复用现有 `OpLibOpInterface` / `OpLibMatchDescriptor` 语义，并保持 deterministic hard-fail 边界。

#### Scenario: Memref-world fusion_region still reuses interface metadata

- **WHEN** `PTOInstantiateAndLowerToLibCall` 处理一个已经过 `PTOViewToMemref` 的 `pto.fusion_region`
- **THEN** 它 MUST 继续通过 `OpLibOpInterface` / `OpLibMatchDescriptor` 构造 template selection 所需信息
- **AND** MUST NOT 为 memref-world region path 引入与既有 single-op/grouped path 不一致的第二套 matcher 协议

#### Scenario: Partially-supported region fails deterministically

- **WHEN** 一个 `pto.fusion_region` 同时包含 active grouped OpLib lowering scope 内的 compute op 与当前不支持的 compute op
- **THEN** `PTOInstantiateAndLowerToLibCall` MUST 发出确定性错误并失败
- **AND** MUST NOT 只改写其中可 lowering 的子集，同时把 unsupported PTO compute op 留在同一 region 中继续流向后续 pass

### Requirement: `trowexpandmul` MUST be active in grouped OpLib lowering for tile fusion regions

为推进 tile fusion 到 LowerToLibCall，本 change 的首批 prelude lowering scope MUST 覆盖 `trowexpandmul`。

#### Scenario: `trowexpandmul` participates in region-based grouped lowering

- **WHEN** 一个 `pto.fusion_region` 中出现 `trowexpandmul -> trowexpandmul -> tadd` 这类当前热点前导链路
- **AND** 该 region 其余 compute op 也都属于 active grouped OpLib lowering scope
- **THEN** `PTOInstantiateAndLowerToLibCall` MUST 把这些 compute op 全部改写到 OP-Lib path
- **AND** MUST NOT 仅因为 `trowexpandmul` 处于 chain prelude 位置就把它保留为 raw PTO compute op

#### Scenario: Row-broadcast semantic roles are preserved for `trowexpandmul`

- **WHEN** grouped lowering 为 `trowexpandmul` 构造 template match / variant selection 请求
- **THEN** lowering MUST 保留哪个 operand 是 full-tile source、哪个 operand 是 row-broadcast source 的语义角色
- **AND** MUST NOT 在 matcher 入口处把这两个角色塌缩成无方向的普通 binary tile-tile 匹配

