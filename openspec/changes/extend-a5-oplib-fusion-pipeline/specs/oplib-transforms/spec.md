# OpLib Transforms Specification

## ADDED Requirements

### Requirement: Fusion group marking and outlining for mixed elementwise chains

PTOAS 的 A5 Level3 OP-Lib transform pipeline MUST 允许连续依赖的 mixed elementwise chain 形成 fusion group，并保持现有 helper/call boundary 模型。

#### Scenario: Mixed tile-tile and tile-scalar chain can form one fusion group

- **WHEN** 连续依赖链仅由以下 12 个 op 组成：
  - `tmul/tdiv/tadd/tsub/tmax/tmin`
  - `tmuls/tdivs/tadds/tsubs/tmaxs/tmins`
- **AND** 后继 op 的某个 tile 输入消费前驱 op 的 tile dst
- **THEN** marking/outlining MUST 允许这些 op 进入同一个 fusion group，即使链中同时包含 tile-tile 与 tile-scalar stage

#### Scenario: Scalar external input does not break mixed chain grouping

- **WHEN** mixed chain 中存在 tile-scalar stage
- **THEN** 其 scalar operand MUST 作为 external input 进入 outlined helper interface，而不是打断当前 fusion group

#### Scenario: Group boundary remains conservative

- **WHEN** 链中出现非目标 op、region 边界、同步/side-effect 语义无法证明兼容的 op，或无法证明当前 op 消费前一 op 的 tile dst
- **THEN** transform MUST 在该处打断 group，并保持保守行为

#### Scenario: Outlined helper preserves existing external ABI model

- **WHEN** 一个 mixed chain 被 outline 成 `__pto_fused_group_*` helper
- **THEN** helper MUST 继续使用现有 private helper + caller single-call boundary 模型
- **AND** MUST NOT 引入新的用户可见 CLI 或公开 IR 入口

### Requirement: Vec-scope-aware low-level fusion

`PTOLowLevelLoopFusion` MUST 直接识别由 OP-Lib inline 产生的相邻 `pto.simd.vec_scope` stage，而不再只匹配裸相邻 `scf.for`。

#### Scenario: Adjacent vec-scope stages collapse into one vec-scope and one loop nest

- **WHEN** `__pto_fused_group_*` helper 中的相邻 stage 具有一致的 loop header、lane mask 构造和允许的 side-effect 形态
- **THEN** `PTOLowLevelLoopFusion` MUST 将这些 stage 合并为单个 `pto.simd.vec_scope` 与单个 loop nest

#### Scenario: Non-canonical helper remains unchanged

- **WHEN** helper 不满足当前支持的 canonical shape，例如 loop header 不一致、mask 构造不一致或 stage body 超出允许的 vector/arith/memref/scf 核心序列
- **THEN** pass MUST 保守保持原 IR，不得做错误合并

### Requirement: Intermediate memory traffic elimination inside fused helper

对 fused helper 链内仅被后继 stage 消费的中间 tile，transform MUST 做 store-to-load forwarding，避免保留多余 round-trip 访存。

#### Scenario: Internal stage result does not round-trip through memory

- **WHEN** 某个 stage 先把中间 tile 写回 `vector.maskedstore`，后继 stage 又立即从同一 tile `vector.maskedload`
- **AND** 该中间 tile 只在当前 fused chain 内被消费
- **THEN** transform MUST 消除该 round-trip，直接让后继 stage 复用前序 stage 的 vector SSA 值

#### Scenario: Final observable outputs keep required stores

- **WHEN** store 对应 group 最终输出，或该 destination 结果在链外仍需可见
- **THEN** transform MAY 保留该最终 `vector.maskedstore`
- **AND** MUST NOT 因链内访存消除而删除该可观察写入
