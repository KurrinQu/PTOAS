# Features Research

## Scope

Features here mean project capabilities the new backend must provide, not end-user product features.

## Table Stakes

These are the minimum capabilities required for the v1 backend to be meaningful.

### Backend Integration

- **Replace the `emitc` backend slot without breaking the existing pass pipeline**
  Complexity: Medium
  Dependencies: New pass registration, new legal/illegal dialect configuration, final output mode.

- **Preserve existing PTO operation semantics**
  Complexity: High
  Dependencies: PTO instruction template references, current op verification rules, shape/layout constraints.

### A5VM IR Model

- **Define legal 256-byte vector types**
  Complexity: Low
  Dependencies: Element-type-specific lane calculation and verification.

- **Define the minimum `a5vm` ops needed for `Abs`**
  Complexity: Medium
  Dependencies: PTO lowering decisions, textual intrinsic emission strategy.

- **Represent builtin/intrinsic variants as explicit op attributes or naming inputs**
  Complexity: Medium
  Dependencies: Local builtin wrapper families such as `vldsx1_*`, distribution kinds, data types.

### PTO Interface Lowering

- **Lower `TLOAD` into an `a5vm` load sequence that preserves enough parameters for intrinsic selection**
  Complexity: High
  Dependencies: Tile/global layout compatibility, valid row/column handling, GM/UB pointer extraction.

- **Lower `TABS` into an `a5vm` absolute-value sequence aligned with PTO unary-op behavior**
  Complexity: Medium
  Dependencies: Vector type legality, matching source/destination tile semantics.

- **Lower `TSTORE` into an `a5vm` store sequence**
  Complexity: High
  Dependencies: Source tile domain, layout checks, store distribution and predicate/mask policy.

### Final Emission

- **Emit textual LLVM HIVM intrinsic IR**
  Complexity: Medium
  Dependencies: Stable textual format, intrinsic name synthesis, printable vector types.

- **Produce a required-intrinsic inventory from actual sample lowering**
  Complexity: Low
  Dependencies: Completed `Abs` path.

## Differentiators

These are not required to prove v1, but they would make the design scale better.

- **A generic lowering framework that can add more PTO ops without changing backend architecture**
  Complexity: Medium
  Reason to defer: Helpful, but v1 only needs a narrow slice.

- **Richer `a5vm` op coverage beyond `Abs`**
  Complexity: High
  Reason to defer: Would expand scope beyond the first acceptance case.

- **Structured intrinsic catalog generation from builtin wrappers**
  Complexity: Medium
  Reason to defer: Useful after the first path is working.

## Anti-Features

These are things to deliberately avoid in v1.

- **Hardcoding a one-off `Abs` string emission path with no reusable framework**
  Why avoid it: It would satisfy the sample but fail the goal of replacing library-based backend lowering cleanly.

- **Re-implementing all PTO operations before proving the backend architecture**
  Why avoid it: High scope risk with little v1 validation benefit.

- **Changing PTO semantics to fit easier codegen**
  Why avoid it: Directly conflicts with the user’s compatibility requirement.

## Dependency Notes

- `TABS` depends on `TLOAD` producing a legal vector-typed representation.
- `TSTORE` depends on how `a5vm` models vector values, memory base, offsets, and possibly masks.
- Intrinsic inventory cannot be finalized until the `Abs` lowering path is implemented and inspected.
