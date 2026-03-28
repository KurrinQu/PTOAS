## ADDED Requirements

### Requirement: Compare/select canonical mask form SHALL preserve A5 lane semantics

对于 `tcmp`、`tcmps`、`tsel`、`tsels` family，OpLib 模板 MAY 使用 byte-per-element mask 作为内部 canonical form，但该表示 MUST 保留与 A5 packed predicate 等价的 lane 级语义。

#### Scenario: Compare templates materialize canonical byte masks

- **WHEN** `tcmp` 或 `tcmps` 通过 OpLib template 产生 mask tile，而模板内部未直接使用 A5 packed predicate 存储形式
- **THEN** active lane 的 false MUST 编码为 `0`，true MUST 编码为 nonzero，且 tail / inactive lane MUST NOT 对后续选择结果产生可观察影响

#### Scenario: Select templates consume canonical byte masks

- **WHEN** `tsel` 或 `tsels` 通过 OpLib template 消费 mask tile
- **THEN** 模板与 lowering MUST 以 `0 == false`、`nonzero == true` 解释 active lane，并且 MUST NOT 在模板边界要求调用方提供 A5 packed predicate 内存布局

### Requirement: Direction-sensitive family canonicalization SHALL preserve external operand semantics

对于 `tile_scalar`、`broadcast_row_binary` 及类似的 direction-sensitive family，OpLib 内部模板 MAY 对 scalar、broadcast operand 或 operand order 做 canonicalization，但 MUST 先保留外部语义角色，再进行模板归一化。

#### Scenario: Tile-scalar direction survives template normalization

- **WHEN** 某个 tile-scalar op 在同级 `pto-isa` 中区分 `tile op scalar` 与 `scalar op tile` 语义，例如 `TDIVS`
- **THEN** OpLib family matching 与模板选择 MUST 在归一化为 `vector.splat` 或等价向量形式之前保留该方向信息，且 MUST NOT 因模板内部统一写法而丢失语义差异

#### Scenario: Row-broadcast families do not narrow A5-legal operand layouts

- **WHEN** 某个 row-broadcast binary op 在同级 `pto-isa` 中允许任一输入与 `dst` 同 shape，另一输入满足 row-broadcast 约束，例如 `trowexpanddiv`、`trowexpandsub`
- **THEN** OpLib family contract MUST 能区分“与 `dst` 同 shape 的 operand”和“row-broadcast operand”，而不是把 `src0` / `src1` 的固定模板顺序当作唯一合法外部输入形态

### Requirement: Distinct reduction variants SHALL map to distinct implemented semantics

如果 OpLib 对外暴露多个 reduction variant，则每个 implemented variant MUST 对应可验证的独立语义；不得继续保留名字不同但实现上无差异的 variant。

#### Scenario: `reduce_colsum(binary)` keeps a binary-specific contract

- **WHEN** `tcolsum` 在 OpLib manifest / template registry 中继续暴露 `variant_id=binary`
- **THEN** 该 variant MUST 具有与 `linear` 不同的 template contract，至少显式体现 `tmp` 参与和 staged accumulation 语义，且 MUST NOT 仅复用 generic linear accumulation skeleton

#### Scenario: Unimplemented binary semantics are not exposed as implemented variants

- **WHEN** PTOAS 不能为 `tcolsum` 或其他 reduction family 提供与 A5 对齐的 `binary` variant 语义
- **THEN** manifest、template metadata 与 lowering MUST 移除、defer 或拒绝该 variant，而不是静默将其映射到 `linear` 模板
