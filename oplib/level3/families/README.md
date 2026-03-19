# Family DSL

This directory holds the declarative Family DSL for A5 OpLib V1 authoring.

- `a5_oplib_v1_family_dsl.json`
  - checked-in family schema data
  - covers family identity, parameter roles, dtype axis, variant axis,
    metadata, matcher keys, and snippet contract binding
- `a5_oplib_v1_snippet_contracts.json`
  - checked-in Mixed-Body MLIR snippet contract data
  - fixes snippet-visible SSA names, result SSA naming, and generator-owned
    responsibilities for binary, tile-scalar, unary, ternary, compare,
    select, reduction, and broadcast families
- `snippets/`
  - checked-in Mixed-Body MLIR snippet sources referenced from active family ops
  - authors maintain vector-body fragments here; generator injects them into the
    skeleton-owned loop/load/store scaffolding
- `../family_dsl.py`
  - loader, validator, snippet-contract checker, and catalog-projection helpers
