# Tile Fusion Store Elision Specification

## Purpose
TBD - created by archiving change close-tile-fusion-store-elision-gaps. Update Purpose after archive.

## Requirements
### Requirement: Fusion-region store elision MUST run before flatten while frontier is still explicit

冗余 store 消除 MUST 发生在 `pto.fusion_region` / `pto.yield` frontier 仍然存在的阶段，而不是等到 flatten 之后再从普通父 block SSA use 反推。

#### Scenario: Store elision consumes explicit fusion-region frontier

- **WHEN** tile fusion 主线已经完成 grouped lowering、inline 和 low-level loop fusion
- **THEN** store-elision MUST 在 `PTOFlattenFusionRegionPass` 之前运行
- **AND** MUST 直接消费 `pto.fusion_region` / `pto.yield` 的显式 yielded frontier

### Requirement: Fusion-region store elision MUST remove non-escaping tail stores

对 lowering 后仅在当前 `pto.fusion_region` 内有效、且生命周期已经结束的 tile 写入，transform MUST 删除其最终物理 store，即使后续已经不存在显式的同 base reload。

#### Scenario: Region-internal tail store is eliminated without requiring a later reload

- **WHEN** 某个 lowered tile 写实例在 `pto.fusion_region` 内完成最后一次写入
- **AND** 该写实例不属于 yielded frontier，且没有其他硬外部可观察 use
- **THEN** store-elision MUST 删除与该写实例对应的最终 `a5vm.vsts` / 等价 store
- **AND** MUST NOT 以“后面还要有同 base `a5vm.vlds`”作为唯一删 store 前提

### Requirement: Fusion-region store elision MUST preserve required frontier materialization

对确实属于 region 对外正式结果的 yielded frontier，store-elision MUST 保守保留必须的物理写入。

#### Scenario: Yielded frontier keeps final store

- **WHEN** 某个 tile 仍然出现在 `pto.yield` / `pto.fusion_region` result frontier 中
- **THEN** store-elision MUST 保留其必要的最终 store
- **AND** MUST NOT 因 region 内的 dead-store elimination 而删除该可观察写入

### Requirement: Yielded frontier consumed through local boundaries MUST stay conservative in v1

对在 region 外继续经过 `pto.treshape` 等 local boundary 的 yielded tile，v1 contract MUST 仍按普通 yielded frontier 保守处理；在没有专门 boundary-aware 方案前，store-elision MUST NOT 激进删除对应 store。

#### Scenario: Yielded tile consumed through treshape remains materialized in v1

- **WHEN** 某个 yielded tile 在 region 外只先经过 `pto.treshape` 等 local boundary，再被后续链路消费
- **THEN** store-elision MUST 仍将其视为 yielded frontier
- **AND** 在缺少专门 boundary-aware forwarding 证明时 MUST 保守保留对应 store
