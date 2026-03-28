## Context

`TileBufType::parse` and `TileBufType::print` in `lib/PTO/IR/PTOTypeDefs.cpp` currently implement a fixed-order verbose keyed syntax. This makes IR dumps hard to scan because identical default-heavy tile types dominate block arguments, operation signatures, and yields.

The current implementation already has all semantic information needed to print more compactly:
- base shape from `shape`
- valid shape from `validShape`
- default config from `TileBufConfigAttr::getDefault`
- normalized memory space from `memorySpace`
- parser target arch via `ScopedPTOParserTargetArch`

This change is limited to textual assembly syntax. It intentionally does not change `TileBufType` storage, verifier rules, memory-space semantics, or codegen behavior.

## Goals / Non-Goals

**Goals:**
- Make canonical `!pto.tile_buf` textual IR substantially shorter and easier to read.
- Preserve strict backward compatibility for existing verbose keyed syntax.
- Define one canonical compact print form to reduce diff noise.
- Keep parser behavior aligned with existing type normalization, including arch-sensitive `loc=left` handling.

**Non-Goals:**
- No change to tile semantics or type invariants.
- No `OpAsmDialectInterface` aliasing in this change.
- No CLI flag or alternate verbose printer mode in this version.
- No changes to CAPI, Python builders, or lowering passes.

## Decisions

### 1. Canonical syntax uses positional `loc` plus shaped base type
Decision:
Use canonical syntax of the form:

`!pto.tile_buf<loc, rowsxcolsxdtype[, keyed-suffix...]>`

Examples:
- `!pto.tile_buf<vec, 1x16xf32>`
- `!pto.tile_buf<vec, 16x128xf32, valid=16x1, blayout=col_major>`

Rationale:
This matches existing MLIR shaped-type conventions already used by other PTO types such as `tensor_view` and `tile`.

Alternative considered:
Use issue-text spacing style such as `1 x 16 x f32`.
Rejected because it is less aligned with existing MLIR type spelling and less natural to implement with current shaped-type helpers.

### 2. Canonical printer elides only derived or default fields
Decision:
The printer omits:
- `valid` when `validShape == shape`
- config fields when equal to `TileBufConfigAttr::getDefault(...)`

The printer emits keyed suffixes only when needed, in this order:
1. `valid`
2. `blayout`
3. `slayout`
4. `fractal`
5. `pad`

Rationale:
This keeps canonical output deterministic and minimizes diff churn.

Alternative considered:
Preserve field names `v_row` and `v_col` in compact mode.
Rejected because `valid=<r>x<c>` is shorter and maps naturally to the 2-D valid-shape concept.

### 3. Parser accepts both legacy and compact syntax
Decision:
`TileBufType::parse` is extended to branch between:
- legacy fixed-order keyed syntax
- compact positional syntax with optional keyed suffixes

Rationale:
Backward compatibility is a hard requirement from the issue and avoids mass test rewrites as a prerequisite for landing the feature.

Alternative considered:
Breaking parser migration to compact syntax only.
Rejected because it would force churn across checked-in tests and samples without semantic benefit.

### 4. Compact syntax remains semantically conservative
Decision:
The compact base shape remains `rowsxcolsxdtype` with static non-negative shape dimensions. Dynamic markers such as `?` are only permitted in `valid`, not in the base shape.

Rationale:
The existing `TileBufType` parser and verifiers treat base tile shape as concrete shape metadata. This change should not widen the semantic space of legal tile buf types.

### 5. Canonical printing reflects normalized type state
Decision:
The printer continues to print the final semantic `TileBufType`, including any parser-time normalization such as `loc=left` behavior dependent on parser target arch.

Rationale:
Canonical printers should reflect actual stored type state, not original input spelling. This preserves round-trip stability and avoids hidden semantic differences.

### 6. Alias support is deferred
Decision:
Do not add `OpAsmDialectInterface` aliasing in this change.

Rationale:
Alias printing is a separate readability feature with different trade-offs around scoping, determinism, and IR familiarity. The current change already provides a meaningful readability gain and should stay narrow.

## Risks / Trade-offs

[Risk] Compact parsing logic becomes harder to maintain than the current linear keyed parser.  
Mitigation: Keep legacy and compact parsing paths explicit and small, and centralize common semantic construction after parse.

[Risk] Canonical printer output changes will require many test expectation updates.  
Mitigation: Limit the blast radius to parse/print affected tests and use canonical examples in new coverage.

[Risk] `loc=left` normalization may create confusing expectations if users omit `blayout`.  
Mitigation: Add explicit A3/A5 tests that show canonical compact output after arch-sensitive normalization.

[Risk] Some malformed compact syntaxes may accidentally parse as valid legacy syntax.  
Mitigation: Parse compact syntax through a distinct positional entry path and add negative tests for ambiguous cases.

## Migration Plan

- Land parser support for compact syntax and keep legacy syntax accepted.
- Switch canonical printing to compact syntax immediately.
- Update parser/printer-sensitive tests to match compact canonical output.
- Add focused compatibility tests proving legacy syntax still parses.
- Defer alias support to a follow-up change if readability remains insufficient.

## Open Questions

- None for v1. Scope, canonical syntax, compatibility mode, and change naming are all locked.
