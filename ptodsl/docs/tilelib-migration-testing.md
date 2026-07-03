# PTODSL TileLib Migration Test Checklist

This page tracks the tests used while migrating PTOAS TileLib expansion from
the legacy TileLang implementation to PTODSL. Run commands from the repository
root.

## Environment

Set up PTOAS, PTODSL, MLIR, and LLVM test-tool paths:

```bash
export PTOAS_ENV_SKIP_SMOKE_TEST=1
source scripts/ptoas_env.sh
export FILECHECK="$LLVM_BUILD_DIR/bin/FileCheck"
```

The Python-only tests do not require rebuilding PTOAS. Tests that invoke
`ptoas` must use a binary rebuilt after the corresponding C++ or TableGen
changes.

## Milestone coverage

| Milestone | Test | Purpose |
|---|---|---|
| Legacy baseline | `expand_tile_op_tilelang_tsub.pto` | Confirms the default TileLang backend still works |
| PTODSL TileLib package | `test_tilelib_catalog.py`, `test_tilelib_constraints.py`, `test_tilelib_elementwise.py`, `test_tilelib_render.py`, `test_tilelib_select.py` | Covers the ported template catalog, legality constraints, template registration and selection, and rendering |
| PTODSL daemon | `test_tilelib_daemon.py` | Covers the Unix-socket protocol, scalar/tile operand specs, metadata, rendering, candidate IDs, and caching |
| PTOAS daemon selection | `expand_tile_op_ptodsl_tsub.pto` | Confirms `--tile-lib-backend=ptodsl` starts and uses the PTODSL daemon |
| Separate metadata/render passes | `expand_tile_op_ptodsl_tadd.pto` | Confirms `InsertTemplateAttributes` records compact metadata before `ExpandTileOp` renders |
| Multi-candidate fallback | `expand_tile_op_ptodsl_tadd.pto` | Confirms `ExpandTileOp` renders candidate index zero when several candidates remain |

## Python TileLib tests

Run every Python TileLib test:

```bash
python3 -m unittest discover -s ptodsl/tests -p 'test_tilelib_*.py'
```

Run the layers individually:

```bash
python3 ptodsl/tests/test_tilelib_constraints.py
python3 ptodsl/tests/test_tilelib_catalog.py
python3 ptodsl/tests/test_tilelib_elementwise.py
python3 ptodsl/tests/test_tilelib_render.py
python3 ptodsl/tests/test_tilelib_select.py
python3 ptodsl/tests/test_tilelib_daemon.py
```

Each command prints `OK` when successful.

The current PTODSL TileLib catalog intentionally covers only ops registered in
`ptodsl.tilelib.templates`. The table-driven catalog test is the scalable place
to add one representative render case per newly ported TileOp.

## PTOAS integration tests

Run the focused lit tests through the generated site configuration. Start lit
from `build/test/lit`; passing source files under `test/lit` directly bypasses
the generated LLVM configuration:

```bash
"$LLVM_BUILD_DIR/bin/llvm-lit" -sv build/test/lit \
  --filter='expand_tile_op_(ptodsl_tsub|ptodsl_tadd|tilelang_tsub)'
```

### PTODSL positive path: one legal candidate

`pto.tsub` has one legal PTODSL candidate. The test checks that PTOAS expands
it into vector operations:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  --tile-lib-backend=ptodsl \
  test/lit/vpto/expand_tile_op_ptodsl_tsub.pto -o - 2>/dev/null |
"$FILECHECK" test/lit/vpto/expand_tile_op_ptodsl_tsub.pto
```

### PTODSL candidate attributes and multi-candidate fallback

Inspect the compact candidate list inserted before fusion:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-pto-ir \
  --tile-lib-backend=ptodsl \
  --mlir-print-ir-after=pto-insert-template-attributes \
  test/lit/vpto/expand_tile_op_ptodsl_tadd.pto \
  -o /dev/null 2>&1 |
"$FILECHECK" test/lit/vpto/expand_tile_op_ptodsl_tadd.pto \
  --check-prefix=META
```

Confirm that insertion also runs before `FusionPlan` when fusion is enabled:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --enable-op-fusion --emit-pto-ir --tile-lib-backend=ptodsl \
  --mlir-print-ir-before=pto-fusion-plan \
  test/lit/vpto/expand_tile_op_ptodsl_tadd.pto \
  -o /dev/null 2>&1 |
"$FILECHECK" test/lit/vpto/expand_tile_op_ptodsl_tadd.pto \
  --check-prefix=PREFUSION
```

Inspect `ExpandTileOp` immediately after selection and confirm candidate zero
was used:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  --tile-lib-backend=ptodsl \
  --mlir-print-ir-after=pto-expand-tile-op \
  test/lit/vpto/expand_tile_op_ptodsl_tadd.pto \
  -o /dev/null 2>&1 |
"$FILECHECK" test/lit/vpto/expand_tile_op_ptodsl_tadd.pto \
  --check-prefix=SELECT
```

Confirm the selected template expands through the full VPTO pipeline:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  --tile-lib-backend=ptodsl \
  test/lit/vpto/expand_tile_op_ptodsl_tadd.pto -o - 2>/dev/null |
"$FILECHECK" test/lit/vpto/expand_tile_op_ptodsl_tadd.pto \
  --check-prefix=EXPAND
```

### Legacy backend regression

Omitting `--tile-lib-backend` must continue to select TileLang:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  test/lit/vpto/expand_tile_op_tilelang_tsub.pto -o - 2>/dev/null |
"$FILECHECK" test/lit/vpto/expand_tile_op_tilelang_tsub.pto
```

## Reading the result

`FileCheck` is silent when it succeeds. Immediately check its status with:

```bash
echo $?
```

`0` means the check passed. When fusion begins filtering candidates, add
coverage for the filtered array while retaining the index-zero fallback test.
