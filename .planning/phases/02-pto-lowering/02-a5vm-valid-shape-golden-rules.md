# A5VM Valid-Shape Golden Rules

Updated: 2026-03-19

Purpose:

- Record the PTO A5 library behavior for vector unary/binary valid-shape handling.
- Use PTO library code as the only golden rule for backend prechecks and lowering contracts.
- Separate "PTO itself forbids this" from "our lowering is stricter than PTO".

## Source Of Truth

Primary references:

- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TUnaryOp.hpp`
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TAdd.hpp`
- same structural pattern also appears in `TMin.hpp`, `TMax.hpp`, `TMul.hpp`, `TDiv.hpp`

## Golden Rule 1: Unary Vec Ops Require Src/Dst Valid Region Equality

PTO A5 unary path:

- `TUNARY_IMPL` obtains:
  - `dstValidRow = dst.GetValidRow()`
  - `dstValidCol = dst.GetValidCol()`
- then it requires:
  - `dstValidRow == src.GetValidRow()`
  - `dstValidCol == src.GetValidCol()`
- otherwise it hard-fails:
  - `PTO_ASSERT(false, "TUNARY: dstTile validRow/validCol must be consistent with of src.")`

Concrete source:

- `TUnaryOp.hpp:186-194`

Implication:

- For `TABS`, `TEXP`, `TLOG`, `TSQRT`, `TRELU`, and the shared `TUNARY_IMPL` family, PTO A5 does **not**
  choose execution extent by `min(src, dst)`.
- PTO A5 does **not** treat destination valid shape as a loose write bound.
- PTO A5 requires unary vec src/dst valid region equality before lowering to builtin loops.

## Golden Rule 2: TRSQRT Uses The Same Equality Rule

`TRSQRT` is not routed through the generic unary instruction wrapper, but its outer contract is the same:

- `TRSQRT_IMPL` loads `dstValidRow/dstValidCol`
- requires equality with `src.GetValidRow()/GetValidCol()`
- otherwise hard-fails:
  - `PTO_ASSERT(false, "TRSQRT: dstTile validRow/validCol must be consistent with of src.")`

Concrete source:

- `TUnaryOp.hpp:173-182`

Implication:

- `TRSQRT` is not an exception to the valid-shape equality rule.

## Golden Rule 3: Unary Loop Shape Selection Depends On Static Tile Form, Not On Relaxed Runtime Shape Reconciliation

Inside `TUnaryOp`:

- if both `DstTile::ValidCol == DstTile::Cols` and `SrcTile::ValidCol == SrcTile::Cols`
  - PTO selects the 1D vector path
- otherwise
  - PTO selects the 2D row-by-row path

Concrete source:

- `TUnaryOp.hpp:102-115`

Implication:

- The loop strategy is chosen from tile form properties.
- Runtime `validRow/validCol` feed the execution extent inside the chosen strategy.
- This is separate from the precondition that src/dst valid region must already match.

## Golden Rule 4: Binary Vec Ops Also Require Valid Region Equality Against Dst

Representative PTO A5 binary path (`TADD_IMPL`):

- `TAddCheck` reads:
  - `validRows = dst.GetValidRow()`
  - `validCols = dst.GetValidCol()`
- then requires:
  - `src0.GetValidRow() == validRows && src0.GetValidCol() == validCols`
  - `src1.GetValidRow() == validRows && src1.GetValidCol() == validCols`
- otherwise hard-fails

Concrete source:

- `TAdd.hpp:53-66`

Same structural rule exists in:

- `TDiv.hpp`
- `TMax.hpp`
- `TMin.hpp`
- `TMul.hpp`

Implication:

- Current PTO A5 vector binary helpers also use strict valid-region equality.
- Any relaxation in PTOAS backend lowering would diverge from PTO unless justified by a higher-level transform that changes the seam IR first.

## Current Backend Comparison

Current PTOAS unary precheck:

- `lib/PTO/Transforms/PTOToA5VMLowering.cpp`
- `checkGenericUnaryContract(...)`
- rejects when:
  - `contract.validRows != dstRows || contract.validCols != dstCols`
- error text:
  - `lowering requires matching source and destination valid region`

Concrete source:

- `PTOToA5VMLowering.cpp:1438-1459`

Conclusion:

- The current unary valid-region equality precheck in PTOAS is aligned with PTO A5 library behavior.
- The currently observed `TRELU` failure is therefore **not** evidence that PTOAS is stricter than PTO on this point.

## Interpretation For `test_dynamic_valid_shape`

Sample:

- `test/samples/Sync/test_dynamic_valid_shape.pto`

Current seam shape pattern:

- source tile valid shape:
  - rows = `1`
  - cols = dynamic `%2` (`32` or `33`)
- destination tile valid shape:
  - rows = `1`
  - cols = static `32`
- operation:
  - `pto.trelu ins(%srcTile) outs(%dstTile)`

Implication under PTO golden rules:

- This shape relation is invalid for the A5 unary helper path as currently implemented in PTO.
- If this sample is expected to compile successfully, the fix likely does **not** belong in simply relaxing unary backend prechecks.

More PTO-aligned candidate directions:

1. Adjust the seam before backend lowering so src/dst valid shapes match.
2. Identify whether the intended PTO library behavior for this case actually uses another helper pattern instead of direct `TRELU`.
3. Prove that seam IR is missing a transformation that the legacy EmitC/PTO path relied on before reaching the unary helper contract.

## Decision Constraint For Future Code Changes

Until proven otherwise by PTO source:

- do not relax unary/binary valid-shape equality checks in the A5VM backend merely to make a sample pass
- first explain how the resulting behavior would still correspond to PTO A5 library semantics
- if the real fix is a seam normalization or a different helper mapping, implement that instead
