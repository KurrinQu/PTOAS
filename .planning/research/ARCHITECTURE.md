# Architecture Research

## Proposed Architecture

## 1. Existing High-Level IR

The Python sample builds PTO IR:

- tensor views from pointers
- partition views
- tile allocation
- `pto.tload`
- `pto.tabs`
- `pto.tstore`

This layer should remain unchanged for v1.

## 2. PTO Semantic Lowering Layer

Introduce a new lowering pass at the current backend boundary where `PTOToEmitC` is used today.

Responsibilities:

- consume PTO ops that are backend-facing
- preserve PTO template-like selection logic
- materialize backend-specific `a5vm` operations
- leave general control flow and scalar ops in shared MLIR dialects

This layer should contain helpers such as:

- `lowerTLOAD`
- `lowerTABS`
- `lowerTSTORE`
- shared utilities for:
  - vector type construction
  - shape/layout legality checks
  - extraction of row/column/stride information
  - intrinsic variant selection inputs

## 3. A5VM Dialect Layer

`a5vm` should be the hardware-facing intermediate representation.

Recommended responsibilities:

- model legal vector values explicitly
- model hardware-facing vector memory operations and arithmetic ops
- encode the information needed to derive textual HIVM intrinsic names

Recommended initial op set for v1:

- load-like op for vector register materialization from memory
- unary abs op over legal `a5vm` vector types
- store-like op for writing vector data back to memory

Recommended initial type/attribute model:

- vector types constrained to fixed 256-byte width
- attributes for load/store distribution or mode where needed
- enough metadata to distinguish intrinsic families such as `vld` vs `vlds`

## 4. Final Textual LLVM HIVM Emission Layer

Lower `a5vm` into textual LLVM-style IR rather than C++.

Responsibilities:

- map `a5vm` vector types to textual LLVM vector type spellings
- synthesize intrinsic names from op kind + element type + lane count + variant
- emit textual operations in a form suitable for downstream external validation

This layer should be isolated so future work can replace textual emission with real LLVM intrinsic ops if/when the environment supports them.

## Component Boundaries

### PTO Lowering Knows

- PTO tile/global semantics
- which PTO interfaces dispatch to which backend families
- how to preserve template-driven variant decisions

### A5VM Knows

- legal vector width
- operation families
- variant attributes that affect intrinsic spelling

### Final Emitter Knows

- textual syntax rules
- intrinsic naming convention
- type printing

## Suggested Build Order

1. Define `a5vm` dialect, types, and minimum ops.
2. Add backend pass plumbing that can replace `emitc` at the existing pass boundary.
3. Implement `lowerTLOAD`, `lowerTABS`, and `lowerTSTORE` for the `Abs` path.
4. Implement `a5vm` textual HIVM emission.
5. Run `Abs` through the path and extract the required intrinsic inventory.

## Architectural Risk

The main risk is collapsing PTO template behavior too early into over-simplified `a5vm` ops. The design should keep enough structured information in the PTO-to-A5VM layer that new intrinsic variants can be added without rewriting the whole backend.
