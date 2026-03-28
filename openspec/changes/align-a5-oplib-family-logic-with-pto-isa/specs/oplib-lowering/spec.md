## ADDED Requirements

### Requirement: OpLib lowering SHALL retain family semantic roles before template normalization

`PTOLowerToOpLibCalls` 在为 A5 OpLib family 构造 `MatchRequest` 或等价 descriptor 时 MUST 先保留 family 语义角色，再允许模板层做 operand swap、`vector.splat` 或 broadcast canonicalization。

#### Scenario: Lowering preserves scalar direction for tile-scalar families

- **WHEN** lowering 处理 direction-sensitive tile-scalar op，例如 `tdivs`
- **THEN** descriptor MUST 保留 scalar 所在位置或等价语义角色，直到模板 / variant 选择完成，且 MUST NOT 因统一的 `vector.splat` 骨架而把不同方向误判为同一 family 形态

#### Scenario: Lowering preserves full-tile vs row-broadcast roles

- **WHEN** lowering 处理 row-broadcast binary op，例如 `trowexpandmul`、`trowexpanddiv`、`trowexpandsub`
- **THEN** descriptor MUST 能标识哪个 operand 与 `dst` 同 shape、哪个 operand 满足 row-broadcast 约束，并据此选择正确 family / variant，而不是仅按固定的 `src0` / `src1` 模板角色匹配

### Requirement: OpLib lowering SHALL hard-fail on collapsed family or variant semantics

若当前 OpLib template registry 无法提供与 spec 对齐的 family / variant 语义，lowering MUST 发出确定性错误或将其排除在 OpLib path 外，不得静默选择语义较弱的近似模板。

#### Scenario: `binary` reduction cannot silently fall back to `linear`

- **WHEN** lowering 处理 `tcolsum` 的 `isBinary=true` 路径，但 template registry 中不存在满足 binary-specific contract 的候选模板
- **THEN** lowering MUST 失败或将该路径保持为 deferred / non-OpLib 处理，而 MUST NOT 静默选择 `variant_id=linear` 模板

#### Scenario: Select masks outside the approved contract are rejected

- **WHEN** lowering 试图将 `tsel` 或 `tsels` 映射到采用 byte-mask canonical form 的 OpLib template
- **THEN** lowering MUST 能证明该 mask 输入满足 compare/select family 定义的 byte-mask contract；若无法证明，则 MUST 失败或退出 OpLib path，而不得把任意整数 tile 视为合法 select mask
