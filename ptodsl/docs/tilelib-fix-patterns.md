# PTODSL TileLib Fix Patterns

This note is an implementation playbook for the `mani/ptodsl` migration work.
It is not a user manual. The goal is to record the common failure shapes we
have already seen, how to recognize them quickly, and what kinds of fixes have
actually worked.

## Why keep this

Many ST failures look similar at first:

- `ExpandTileOp requires at least one template candidate`
- `NoMatchingTemplate`
- `custom constraints are not satisfied`
- build passes, but compare fails

In practice, these come from a small number of recurring root causes. Writing
them down helps avoid re-debugging the same pattern from scratch.

## Fast classification

When a testcase fails, classify it first:

| Failure shape | Usual meaning | First place to look |
|---|---|---|
| `dtype signature ... is not supported` | template exists, but the registered dtype matrix is too narrow | tile template `dtypes=` list and any dtype-specific body logic |
| `custom constraints are not satisfied` | template exists, but legality is too narrow for ST operand layout / memory-space / valid-shape form | template constraint helpers |
| `requires at least one template candidate` / `no candidate survives` | candidate was never inserted, was dropped by a pass, or selection key changed shape | daemon metadata path, `InsertTemplateAttributes`, `PTOViewToMemref`, `ExpandTileOp` |
| build succeeds, compare fails | semantics differ from TileLangDSL or ST golden | template body logic, store offsets, padding, reduction loop shape, precision path |
| smoke passes, non-smoke fails | usually a missing dtype/version/path not exercised by smoke, or a large-shape / high-precision semantic gap | non-smoke-only ST cases and legacy template body |
| PTODSL tracing error | template mixes Python compile-time control with runtime SSA values | PTODSL-authored control flow, scalar coercion, loop structure |

## Common problem patterns and fixes

### 1. Missing dtype/signature coverage

**Symptom**

- `NoMatchingTemplate`
- `dtype signature ('f16', 'f32', 'f16') is not supported`

**What it usually means**

The template is fundamentally right, but PTODSL registered fewer legal
signatures than TileLangDSL/ST actually uses.

**What fixed it**

- extend the template `dtypes=` list
- if needed, add a small dtype-specific path inside the body

**Recent example**

- `tlrelu`
  - smoke passed because it only used `f32`
  - non-smoke used `src=f16, slope=f32, dst=f16`
  - fix: add `("f16", "f32", "f16")` and coerce the runtime slope to `f16`

### 2. Constraint too narrow for the real ST operand form

**Symptom**

- `custom constraints are not satisfied`
- template exists, but ST still cannot select it

**What it usually means**

The template is present, but the accepted memory space, layout, valid shape, or
operand form is narrower than the real ST emission.

**What fixed it**

- compare the ST `.pto` operand form with the PTODSL constraint helper
- relax the legality only to the real used shape/layout combinations
- do not immediately rewrite the body; first make sure selection is correct

**Recent examples**

- `tcmps`
  - initial constraints were too narrow
  - ST used packed predicate destination shapes that did not satisfy the old
    assumptions
- likely next candidates for this pattern:
  - `tsel`
  - `textract`
  - `txors`

### 3. Context attributes reached selection, but not rendering

**Symptom**

- candidate selection succeeds
- expanded template body behaves as if attr-driven mode stayed at the default
- wrong compare mode, round mode, precision mode, etc.

**What it usually means**

The daemon saw `context_attrs` during metadata selection, but the actual
render/specialization step dropped them.

**What fixed it**

- forward `context_attrs` into the template `specialize(...)` call during
  render, not only during metadata selection

**Recent example**

- `tcmp` / `tcmps`
  - `cmp_mode` was effectively rendered as default behavior
  - fix: preserve `context_attrs` all the way into template specialization

### 4. Op attrs lost during `PTOViewToMemref`

**Symptom**

- metadata insertion seems fine upstream
- later `ExpandTileOp` sees no candidates
- op recreation in a transform silently drops attrs

**What it usually means**

One of the view-to-memref rewrites recreated the TileOp but did not preserve
attrs like `candidates` or other mode fields.

**What fixed it**

- replace manual op recreation with the local helper that clones attrs
- audit all rewritten operands of that op at the same time

**Recent example**

- `TCmpSOp` in `PTOViewToMemref.cpp`
  - manual recreation dropped attrs
  - fix: use the cloned-attrs replacement helper

### 5. Callable form mismatch with TileLang/ST

**Symptom**

- template exists
- selection fails or body assumptions do not match actual operand list
- often happens on ops with optional tmp/scalar/extra-vector operands

**What it usually means**

PTODSL encoded a simplified form, but ST emits a different operand order or a
different auxiliary operand shape/dtype.

**What fixed it**

- inspect the ST `.pto` op directly
- compare against the legacy TileLangDSL template parameter order
- fix the PTODSL signature and only then adjust body logic

**Recent examples**

- `tcmps`
- `tmrgsort`
- `tsort32`

### 6. Runtime scalar vs compile-time literal confusion

**Symptom**

- PTODSL tracing error when trying to cast or branch on a scalar kernel
  argument
- constructors like `pto.f16(...)` work for Python literals but fail for runtime
  scalar values

**What it usually means**

The template is treating a runtime scalar SSA value like a compile-time Python
literal.

**What fixed it**

- use PTODSL scalar coercion utilities for runtime values
- do not use constant constructors for runtime scalar adaptation

**Recent example**

- `tlrelu`
  - `pto.f16(slope)` was wrong when `slope` was a runtime kernel argument
  - fix: use runtime scalar coercion to `f16`

### 7. Python control flow mixed with PTODSL runtime values

**Symptom**

- tracing misuse errors like:
  - runtime value used as Python loop bound
  - runtime value used in native `if`

**What it usually means**

The template relies on Python control flow for something that became a device
side value during tracing.

**What fixed it**

- keep branching on compile-time quantities only
- use PTODSL-authored control flow for runtime-dependent decisions
- simplify the loop structure so the split between compile-time and runtime is
  explicit

**Recent example**

- `tsort32`
  - large non-smoke path hit runtime/control-flow trouble in earlier attempts

### 8. Build problem turns into semantic problem after template coverage lands

**Symptom**

- older snapshot showed `NoMatchingTemplate`
- fresh rerun now builds and runs
- compare still fails on a narrow case family

**What it usually means**

This is progress. The issue moved from coverage to behavior.

**What to do**

- do not keep documenting it as a build blocker
- move it to the semantic-parity list
- isolate the exact failing ST case family

**Recent examples**

- `tdivs`
  - no longer a template-selection failure
  - now fails a high-precision subnormal scalar-src compare case
- `tsort32`
  - no longer a no-template failure
  - now fails large unaligned non-smoke semantics

## Reusable debugging workflow

### A. For `NoMatchingTemplate` / `custom constraints are not satisfied`

1. Read the failing `.pto` call form in the ST testcase.
2. Read the TileLangDSL template for the same op.
3. Read the PTODSL template decorator:
   - `dtypes=`
   - constraint helpers
   - parameter order
4. Decide whether the miss is:
   - missing dtype
   - too-narrow constraint
   - wrong callable form
5. Add a small focused catalog/daemon test if the fix is local.

### B. For `no candidate survives`

1. Confirm whether PTODSL has a template for that op at all.
2. Check whether daemon metadata returns candidates.
3. If metadata is correct, inspect attr preservation across:
   - `InsertTemplateAttributes`
   - `PTOViewToMemref`
   - `ExpandTileOp`
4. Diff a working nearby op if possible.

### C. For build-pass but compare-fail

1. Identify the smallest failing ST case.
2. Diff PTODSL body vs TileLangDSL body.
3. Look for:
   - pack/store offsets
   - pad constants
   - reduction loop depth
   - tail handling
   - precision widening/casting
4. Re-run only the affected testcase after each change.

## Patterns that are usually low effort

These tend to be good grouped fixes:

- dtype matrix expansion
- constraint widening to real ST forms
- runtime scalar coercion fixes
- attr forwarding / attr preservation fixes

## Patterns that are usually not low effort

These tend to sprawl:

- true backend candidate-propagation failures in cube paths
- high-precision semantic parity
- wrong-output bugs in reduction/arg-reduction writeback
- large-shape tail handling
- random / sort semantic parity

## Keep this note current

When an ST failure is fixed, add a short line here if it introduced a new
repeatable debugging lesson. The goal is not to list every tileop, but to keep
the small set of recurring fix patterns visible.
