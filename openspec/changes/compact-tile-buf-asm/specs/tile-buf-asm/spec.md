## ADDED Requirements

### Requirement: Tile buf types MUST print in a compact canonical form
The system MUST print `!pto.tile_buf` types in a compact canonical textual form that uses positional `loc` and shaped `rowsxcolsxdtype` syntax. The default printer MUST omit fields whose values are implied by the base shape or the dialect default configuration.

#### Scenario: Fully default tile buf prints without keyed suffix
- **WHEN** a `!pto.tile_buf` has `validShape == shape` and uses the default config
- **THEN** the printer MUST emit `!pto.tile_buf<loc, rowsxcolsxdtype>`
- **AND** the printer MUST NOT emit `valid`, `blayout`, `slayout`, `fractal`, or `pad`

#### Scenario: Non-default valid shape prints a keyed valid suffix
- **WHEN** a `!pto.tile_buf` has `validShape != shape`
- **THEN** the printer MUST emit a `valid=<rows>x<cols>` suffix
- **AND** the printer MUST NOT emit `v_row` or `v_col` in the canonical output

#### Scenario: Non-default config prints only changed fields
- **WHEN** a `!pto.tile_buf` has a non-default config field
- **THEN** the printer MUST emit only the changed config fields as keyed suffixes
- **AND** the suffix order MUST be `valid`, `blayout`, `slayout`, `fractal`, `pad`

### Requirement: Tile buf parser MUST accept both legacy and compact syntax
The system MUST parse both the existing verbose keyed syntax and the new compact syntax for `!pto.tile_buf`. Legacy syntax support MUST remain backward-compatible for existing checked-in `.pto` tests and samples.

#### Scenario: Legacy keyed syntax remains accepted
- **WHEN** the parser reads a `!pto.tile_buf` written as `loc=..., dtype=..., rows=..., cols=..., v_row=..., v_col=..., blayout=..., slayout=..., fractal=..., pad=...`
- **THEN** the parser MUST construct the same semantic `TileBufType` as before this change
- **AND** parsing the type MUST NOT require any migration of existing test inputs

#### Scenario: Compact syntax parses and reprints canonically
- **WHEN** the parser reads a compact tile buf such as `!pto.tile_buf<vec, 16x128xf32, valid=16x1, blayout=col_major>`
- **THEN** it MUST construct the corresponding `TileBufType`
- **AND** printing that type MUST produce the compact canonical form

#### Scenario: Invalid compact syntax is rejected
- **WHEN** the parser reads a malformed compact tile buf
- **THEN** parsing MUST fail with a diagnostic
- **AND** the parser MUST NOT silently reinterpret malformed compact text as another valid type

### Requirement: Compact tile buf assembly MUST preserve existing semantics
The compact syntax is assembly sugar only. It MUST NOT change type invariants, verifier behavior, target-arch-sensitive normalization, or downstream lowering behavior.

#### Scenario: Arch-sensitive left-buffer normalization is preserved
- **WHEN** the parser reads a `!pto.tile_buf` for `loc=left`
- **AND** parser target arch is scoped by the CLI
- **THEN** the same arch-sensitive normalization rules MUST apply as in the legacy syntax
- **AND** canonical printing MUST reflect the normalized type state rather than the user's original textual spelling

#### Scenario: No semantic difference between legacy and compact forms
- **WHEN** the same tile buf is expressed once in legacy syntax and once in compact syntax
- **THEN** both forms MUST parse to equivalent `TileBufType` values
- **AND** all verifier and lowering behavior MUST remain identical
