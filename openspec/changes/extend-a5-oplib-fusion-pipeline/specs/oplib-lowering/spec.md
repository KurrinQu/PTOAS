# OpLib Lowering Specification

## ADDED Requirements

### Requirement: Grouped OP-Lib lowering SHALL preserve single-op coverage for mixed elementwise chains

PTOAS 的 grouped OP-Lib lowering path MUST 对 mixed elementwise chain 保持与 single-op lowering 相同的目标 op 覆盖、descriptor 语义和失败策略。

#### Scenario: Grouped path supports the same 12 in-scope elementwise ops as single-op path

- **WHEN** `PTOInstantiateAndLowerToLibCall` 处理已经形成 fusion group 的 elementwise chain
- **THEN** grouped lowering path MUST 支持以下 12 个 op，与 single-op path 保持一致：
  - `tmul/tdiv/tadd/tsub/tmax/tmin`
  - `tmuls/tdivs/tadds/tsubs/tmaxs/tmins`

#### Scenario: Mixed chain grouped lowering reuses `OpLibOpInterface`

- **WHEN** mixed chain 中同时包含 tile-tile 与 tile-scalar stage
- **THEN** grouped lowering MUST 继续通过现有 `OpLibOpInterface` / `OpLibMatchDescriptor` 构造匹配请求
- **AND** MUST NOT 为 grouped path 引入与 single-op path 不一致的 matcher 协议

#### Scenario: `tdivs` keeps operand-order and variant constraints in grouped lowering

- **WHEN** grouped chain 中出现 `tdivs`
- **THEN** lowering MUST 保留 `operandOrder` 与 `requiredVariantId` 约束
- **AND** MUST 正确覆盖 `tile_scalar` 与 `scalar_tile` 两种语义方向

#### Scenario: Grouped lowering remains hard-fail on lowering mismatch

- **WHEN** grouped path 中某个上述目标 op 在 descriptor 构造、候选选择、实例化或 call rewrite 任一环节失败
- **THEN** pass MUST 发出确定性错误并失败
- **AND** MUST NOT 静默保留原始 PTO op 或仅记录 warning
