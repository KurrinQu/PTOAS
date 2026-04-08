# TileLang DSL Unsupported And Partial Features

## Scope

This document records the gap between the broad language surface described in
`tilelang-dsl-guide.md` and what the current standalone `tilelang_dsl` package
actually implements under:

- `tilelang-dsl/python/tilelang_dsl/`
- `tilelang-dsl/tests/`

Use this file as a quick "what is still missing" index. For the implemented
contract, treat these as the source-of-truth companion documents:

- `v1-surface.md`
- `v1-lowering.md`
- `matcher-and-advanced-surface-migration.md`

## Status Labels

- `Unsupported`: the public surface is documented but not exported or not
  accepted by the frontend at all.
- `Partial`: the concept exists, but only a narrower subset works in the
  current implementation.

## Unsupported Features

### Missing Public Type Constructors And Aliases

The guide documents a richer type-construction surface that is not exported by
the current package:

- `pto.tile(...)`
- `pto.memref(...)`
- `pto.vreg(...)`
- `pto.mask_b8`, `pto.mask_b16`, `pto.mask_b32`
- `GMPtr`, `UBPtr`, `UBRef`
- `BLayout`, `SLayout`, `PadValue`
- `SyncOpType`

Today, the public package exports annotation markers (`TensorView`, `Tile`),
scalar dtypes, `ptr(...)`, `PadMode`, `TileConfig`, matcher APIs, and a small
set of enums. The list above covers the remaining missing public constructors
and aliases from the guide.

### Missing Tile/Tensor Utility Methods

The following guide surfaces are not implemented as public APIs:

- `tile.to_ubref()`
- `tile.as_ptr()`
- `tile.to_memref()`
- `tile.slice(...)`
- `tile.reshape(...)`
- `pto.tile_from_ptr(...)`
- `pto.tile_from_memref(...)`
- `pto.tile_with_strides(...)`
- `pto.tile_config(...)`

### Missing Sync/Buffer Control Ops

These documented surfaces are not accepted by the current frontend:

- `pto.get_buf(...)`
- `pto.rls_buf(...)`

### Missing High-Level Tile Copy Surface

The guide documents `pto.dma_copy(...)`, but the current package does not
provide that high-level operation. Only the raw low-level copy op
`pto.copy_ubuf_to_ubuf(...)` exists in advanced mode.

### Missing Vector Load/Store Families

Only `pto.vlds(...)` and `pto.vsts(...)` are implemented from the guide's
load/store families. The following documented ops are still unsupported:

- `pto.vldas(...)`
- `pto.vldus(...)`
- `pto.vplds(...)`
- `pto.vldx2(...)`
- `pto.vsld(...)`
- `pto.psts(...)`
- `pto.vsst(...)`
- `pto.vstx2(...)`
- `pto.vsta(...)`
- `pto.pstu(...)`
- `pto.vstu(...)`
- `pto.vstus(...)`
- `pto.vstur(...)`

### Missing Direct Predicate Constructor/Compare APIs

The implementation expects users to go through `pto.make_mask(...)` rather than
call the underlying mask ops directly. These guide-documented APIs are not part
of the supported authoring surface:

- `pto.pset_b8(...)`, `pto.pset_b16(...)`, `pto.pset_b32(...)`
- `pto.pge_b8(...)`, `pto.pge_b16(...)`, `pto.pge_b32(...)`
- `pto.plt_b8(...)`, `pto.plt_b16(...)`, `pto.plt_b32(...)`

### Missing Unary/Conversion/Special Vector Ops

The current package supports only a small subset of the unary/conversion
families. These guide-documented ops are still unsupported:

- `pto.vln(...)`
- `pto.vsqrt(...)`
- `pto.vrec(...)`
- `pto.vcadd(...)`
- `pto.vcmax(...)`
- `pto.vbcnt(...)`
- `pto.vtrc(...)`
- `pto.vcvt(...)`
- `pto.vbitsort(...)`
- `pto.vmrgsort4(...)`

### Missing Shift/Leaky-ReLU Vector Ops

These documented ops are not currently supported:

- `pto.vshl(...)`
- `pto.vshr(...)`
- `pto.vlrelu(...)`
- `pto.vshls(...)`
- `pto.vshrs(...)`

### Missing Predicate Rearrangement Shorthands

The guide documents mask-specific rearrangement helpers that are not currently
implemented:

- `pto.pdintlv_b8(...)`
- `pto.pintlv_b16(...)`

### Deferred Surface

`pto.vreduce(...)` is still explicitly deferred and remains rejected even in
`advanced=True` kernels.

## Partial Features

### Scalar Constants And Literal Typing

The guide describes:

- `pto.i32(1024)` / `pto.f32(0.0)` style constructors
- automatic `float -> pto.f32` literal typing

The current implementation does not support those forms. Scalar dtype objects
such as `pto.i32` and `pto.f32` are type markers, not callable constructors.
Literal support is currently limited to:

- `bool`
- `int`
- `str`
- `None`

### TensorView Attribute Model

`TensorView` currently supports only a narrow attribute subset:

- `shape`
- `element_type`
- `valid_shape`

The following documented attributes are not implemented:

- `strides`
- `offset`

In practice, `TensorView` is still treated as a 2D GM view in the current
profile.

### Tile Attribute Model

`Tile` currently supports only a narrow attribute subset in semantic analysis:

- `shape`
- `element_type`
- `valid_shape`

The guide documents additional properties that are not currently supported:

- `memory_space`
- `config`
- `rank`
- `num_elements`
- `valid_elements`
- `layout_descriptor`
- `strides`
- `byte_strides`
- `offset`

### Tile Config Semantics

`TileConfig` can be attached during specialization, but lowering does not yet
honor the rich layout/padding semantics described in the guide. The rendered
tile type is effectively fixed to a hard-coded baseline:

- `blayout=row_major`
- `slayout=none_box`
- `fractal=512`
- `pad=0`

So this is currently metadata storage rather than full behavioral support.

### TensorView Slicing

The guide presents general Python slicing with dynamic starts and strides. The
current stable DMA-oriented implementation is still a narrower 2D profile:

- slice `stop` must be explicit on all dimensions
- slice `start` may be a compile-time constant or runtime index expression
- slice `step` must be a static positive integer
- dimension 0 may use `step > 1`
- dimension 1 must keep `step == 1` (current DMA restriction)

Dynamic bounds are supported within those constraints.

### High-Level DMA Ops

`pto.dma_load(...)` and `pto.dma_store(...)` now accept the documented keyword
surface on the stable path, but the implemented contract is still a narrowed
frontend-only subset.

Implemented today:

- non-zero/dynamic slice starts and dynamic stops
- static positive outer-axis slice steps with inferred DMA strides
- `dma_load` keywords: `pad_mode`, `pad_value`, `left_padding`,
  `right_padding`, `init_out_buffer`
- `dma_store` trim keywords: `left_padding`, `right_padding`
- padded `dma_load` lowering for `PadNull`, `PadFirstElem`, and `PadValue`
- trimmed `dma_store` lowering for `pad_mode=PadMode.PadNull`

Still unsupported or intentionally deferred:

- TensorView slices up to 5D are supported
- the destination/source Tile must be a statically specialized rank-2 UB Tile
- dynamic slice `step`
- stepped DMA on TensorView slice dimension 1
- `dma_load(..., pad_mode=PadMode.PadNull, init_out_buffer=True)`
- `dma_store(..., pad_mode != PadMode.PadNull, ...)`
- `dma_store(..., pad_value=..., ...)`
- store-side GM fill / backend-init semantics
- any shape profile where slice extent does not match the Tile window after
  applying padding or trim

### Tile Indexing Sugar

Tile indexing sugar is partially implemented on the stable authoring path.

Currently supported:

- rank-1: `tile[start:]`
- rank-2: `tile[row, col:]`
- only for `pto.vlds(...)` and `pto.vsts(...)`

Not currently supported from the guide's broader indexing model:

- column-major syntax such as `tile[row_start:, col_index]`
- single-element syntax such as `tile[row, col]` and `tile[pos]`
- explicit slice `stop`
- stepped tile vector slices
- the guide's wider indexed op family (`vldas`, `vldus`, `vplds`, `vldx2`,
  `vsld`, `psts`, `vsst`, `vstx2`, `vsta`)

### Control-Flow Result Merging

The frontend does analyze loop-carried values and merged `if` results, but
lowering still has a hard limit:

- at most one loop-carried binding per loop
- at most one merged `if`/`else` binding per conditional

So the language feature exists conceptually, but multi-value merge cases are
not fully lowered yet.

### Tile Profile Breadth

The guide discusses Tile memory spaces in more general terms, but bare Tile
specialization still only accepts:

- rank-1 or rank-2 Tiles
- static physical shape
- `MemorySpace.UB`

So GM Tiles and more general profiles are not supported yet.

## Currently Implemented Core Surface

For quick orientation, the current package head is strongest in these areas:

- matcher-driven kernel selection
- `templates={...}` and `pto.tpl(...)`
- `ptr(...)`, `pto.castptr(...)`, `pto.addptr(...)`
- low-level DMA config/copy ops
- `pto.make_mask(...)`
- `pto.vlds(...)` and `pto.vsts(...)`
- base unary/binary/vector-scalar vector ops
- advanced compare/select/carry/rearrangement families

If you need the exact supported boundary for implementation work, prefer the
source files and tests over the broader guide:

- `tilelang-dsl/python/tilelang_dsl/support_matrix.py`
- `tilelang-dsl/python/tilelang_dsl/semantic.py`
- `tilelang-dsl/python/tilelang_dsl/lowering.py`
- `tilelang-dsl/tests/test_tilelang_dsl_v1.py`
