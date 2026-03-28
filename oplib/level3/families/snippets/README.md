# Mixed-Body MLIR Snippets

This directory holds author-maintained Mixed-Body MLIR snippets for A5 OpLib V1.

- snippet files are referenced from `a5_oplib_v1_family_dsl.json`
- snippets describe only family-local vector compute fragments
- generator-owned scaffolding remains in `../skeletons/*.tmpl.mlir`

## Section Markers

- files without markers are treated as compute-only snippets
- optional setup sections use:
  - `// SNIPPET_SETUP_BEGIN`
  - `// SNIPPET_SETUP_END`
- optional compute sections use:
  - `// SNIPPET_COMPUTE_BEGIN`
  - `// SNIPPET_COMPUTE_END`

When markers are present, the generator injects setup lines into `@@EXTRA_SETUP@@`
and compute lines into `@@COMPUTE@@`.
