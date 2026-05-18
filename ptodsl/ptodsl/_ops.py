# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
PTO operation wrappers.

Every function in this module emits one or more MLIR operations at the
active insertion point and returns the primary SSA result(s).

Design rules:
- Vector math ops infer the result type from the first operand's type.
- ``vlds`` / ``vbrc_load`` still require an explicit ``vreg_type`` argument
  because the result type cannot be inferred from the pointer alone.
- ``make_tensor_view`` infers the TensorViewType from ``len(shape)`` and the
  pointer's element type.
- ``partition_view`` infers the PartitionTensorViewType from the source type.
"""

from ._bootstrap import make_context  # noqa: F401 – ensure MLIR on sys.path
from ._diagnostics import tile_row_alignment_error
from ._host_tensors import resolve_tensor_data_entry
from ._scalar_coercion import coerce_scalar_to_type, materialize_scalar_literal
from ._runtime_scalar_ops import classify_runtime_scalar_type, emit_runtime_binary_op
from ._surface_values import (
    MaskResultValue,
    PartitionTensorViewValue,
    TensorViewValue,
    TileSliceValue,
    TileValue,
    _unwrap_sequence,
    compose_partition_spec,
    emit_as_ptr,
    infer_tile_element_type,
    parse_tile_type_metadata,
    unwrap_surface_value,
    wrap_surface_value,
)
from ._types import _resolve, mask_type, part_tensor_view_type, tensor_view_type, vreg_type

from mlir.dialects import arith, pto as _pto
from mlir.ir import (
    Attribute,
    BF16Type,
    F16Type,
    F32Type,
    FloatAttr,
    IndexType,
    IntegerType,
    MemRefType,
    Type,
)

# Pipe name shorthands → canonical PIPE_* names
_PIPE_ALIASES = {
    "MTE1": "PIPE_MTE1",
    "MTE2": "PIPE_MTE2",
    "MTE3": "PIPE_MTE3",
    "MTE4": "PIPE_MTE4",
    "V":    "PIPE_V",
    "M":    "PIPE_M",
    "S":    "PIPE_S",
    "ALL":  "PIPE_ALL",
}


def _pipe_attr(name: str):
    if not isinstance(name, str):
        return _pto.PipeAttr.get(name)
    canonical = _PIPE_ALIASES.get(name, name)
    if not canonical.startswith("PIPE_"):
        canonical = "PIPE_" + canonical
    return _pto.PipeAttr.get(getattr(_pto.PIPE, canonical))


def _event_attr(event_id: int):
    return getattr(_pto, f"EVENT_ID{event_id}")


# ── Constants ────────────────────────────────────────────────────────────────

def const(value: int, *, dtype=None):
    """
    Emit an ``arith.constant``.

    ``dtype`` is a ``_DType`` descriptor or a concrete ``mlir.ir.Type``.
    Defaults to ``index`` when omitted.
    """
    from ._types import index as _idx_dtype
    mlir_type = _resolve(dtype) if dtype is not None else _resolve(_idx_dtype)
    return wrap_surface_value(arith.ConstantOp(mlir_type, value).result)


# ── Pointer ops ───────────────────────────────────────────────────────────────

def castptr(int_addr, result_ptr_type):
    """``pto.castptr`` – cast an integer address to a typed PTO pointer."""
    return wrap_surface_value(
        _pto.CastPtrOp(_resolve(result_ptr_type), unwrap_surface_value(int_addr)).result
    )


def addptr(base_ptr, index_offset):
    """``pto.addptr`` – advance a pointer by an index offset."""
    return wrap_surface_value(
        _pto.AddPtrOp(unwrap_surface_value(base_ptr), unwrap_surface_value(index_offset)).result
    )


# ── Vector load / store ───────────────────────────────────────────────────────

def vlds(src_ptr, offset=None, result_vreg_type=None):
    """``pto.vlds`` – vector load from a tile slice or from *src_ptr* at *offset*."""
    if isinstance(src_ptr, TileSliceValue):
        if offset is not None or result_vreg_type is not None:
            raise TypeError("vlds(tile[row, col:]) infers its memref slice and vreg type; do not pass offset/result_vreg_type")
        return wrap_surface_value(_pto.VldsOp(
            _infer_vreg_type_from_tile_slice(src_ptr),
            unwrap_surface_value(src_ptr),
            _index_zero(),
        ).result)

    if offset is None or result_vreg_type is None:
        raise TypeError("vlds(ptr, offset, result_vreg_type) requires both offset and result_vreg_type")
    return wrap_surface_value(_pto.VldsOp(
        _resolve(result_vreg_type),
        unwrap_surface_value(src_ptr),
        unwrap_surface_value(offset),
    ).result)


def vbrc_load(src_ptr, offset, result_vreg_type):
    """``pto.vlds {dist="BRC_B32"}`` – broadcast a scalar into all lanes."""
    return wrap_surface_value(
        _pto.VldsOp(
            _resolve(result_vreg_type),
            unwrap_surface_value(src_ptr),
            unwrap_surface_value(offset),
            dist="BRC_B32",
        ).result
    )


def vsts(val, dst_ptr, offset, mask=None):
    """``pto.vsts`` – vector store to a tile slice or to *dst_ptr* at *offset*."""
    if isinstance(dst_ptr, TileSliceValue):
        if mask is not None:
            raise TypeError("vsts(vec, tile[row, col:], mask) does not accept a separate offset argument")
        _pto.VstsOp(
            unwrap_surface_value(val),
            unwrap_surface_value(dst_ptr),
            _index_zero(),
            unwrap_surface_value(offset),
        )
        return

    if mask is None:
        raise TypeError("vsts(vec, ptr, offset, mask) requires an explicit mask")
    _pto.VstsOp(
        unwrap_surface_value(val),
        unwrap_surface_value(dst_ptr),
        unwrap_surface_value(offset),
        unwrap_surface_value(mask),
    )


def vsts_1pt(val, dst_ptr, offset, mask):
    """``pto.vsts {dist="1PT_B32"}`` – store only the lowest lane."""
    _pto.VstsOp(
        unwrap_surface_value(val),
        unwrap_surface_value(dst_ptr),
        unwrap_surface_value(offset),
        unwrap_surface_value(mask),
        dist="1PT_B32",
    )


# ── Mask / predicate ops ──────────────────────────────────────────────────────

def plt_b32(scalar):
    """
    ``pto.plt_b32`` – predicate-load from a 32-bit scalar.

    Returns ``(mask_value, scalar_out)``.  ``scalar_out`` is often unused
    and can be discarded with ``_``.
    """
    plt_op = _pto.PltB32Op(
        _resolve(mask_type("b32")),
        IntegerType.get_signless(32),
        unwrap_surface_value(scalar),
    )
    return wrap_surface_value(plt_op.mask), wrap_surface_value(plt_op.scalar_out)


def pset_b32(pattern: str):
    """``pto.pset_b32 "PATTERN"`` → ``!pto.mask<b32>``."""
    return wrap_surface_value(_pto.PsetB32Op(_resolve(mask_type("b32")), pattern).result)


# ── Vector math (result type inferred from first operand) ─────────────────────

def vadd(lhs, rhs, mask, result_type=None):
    """``pto.vadd`` – element-wise add."""
    rt = result_type if result_type is not None else lhs.type
    return wrap_surface_value(
        _pto.VaddOp(
            _resolve(rt),
            unwrap_surface_value(lhs),
            unwrap_surface_value(rhs),
            unwrap_surface_value(mask),
        ).result
    )


def vmul(lhs, rhs, mask):
    """``pto.vmul`` – element-wise multiply."""
    return wrap_surface_value(
        _pto.VmulOp(
            unwrap_surface_value(lhs).type,
            unwrap_surface_value(lhs),
            unwrap_surface_value(rhs),
            unwrap_surface_value(mask),
        ).result
    )


def vmax(lhs, rhs, mask):
    """``pto.vmax`` – element-wise maximum."""
    return wrap_surface_value(
        _pto.VmaxOp(
            unwrap_surface_value(lhs).type,
            unwrap_surface_value(lhs),
            unwrap_surface_value(rhs),
            unwrap_surface_value(mask),
        ).result
    )


def vdiv(lhs, rhs, mask):
    """``pto.vdiv`` – element-wise divide."""
    return wrap_surface_value(
        _pto.VdivOp(
            unwrap_surface_value(lhs).type,
            unwrap_surface_value(lhs),
            unwrap_surface_value(rhs),
            unwrap_surface_value(mask),
        ).result
    )


def vcmax(v, mask):
    """``pto.vcmax`` – cross-lane maximum reduction."""
    return wrap_surface_value(
        _pto.VcmaxOp(
            unwrap_surface_value(v).type,
            unwrap_surface_value(v),
            unwrap_surface_value(mask),
        ).result
    )


def vcadd(v, mask):
    """``pto.vcadd`` – cross-lane add (sum reduction)."""
    return wrap_surface_value(
        _pto.VcaddOp(
            unwrap_surface_value(v).type,
            unwrap_surface_value(v),
            unwrap_surface_value(mask),
        ).result
    )


def vdup(v, mask, *, position=None):
    """``pto.vdup`` – duplicate a lane value into all lanes.

    Pass ``position="LOWEST"`` to broadcast the lowest (lane-0) element.
    """
    return wrap_surface_value(
        _pto.VdupOp(
            unwrap_surface_value(v).type,
            unwrap_surface_value(v),
            unwrap_surface_value(mask),
            position=position,
        ).result
    )


def vexpdif(inp, ref, mask, part: str = "ODD"):
    """``pto.vexpdif`` – ``exp(inp - ref)`` selecting ODD or EVEN lanes."""
    return wrap_surface_value(
        _pto.VexpdifOp(
            unwrap_surface_value(inp).type,
            unwrap_surface_value(inp),
            unwrap_surface_value(ref),
            unwrap_surface_value(mask),
            part,
        ).result
    )


def vexp(inp, mask):
    """``pto.vexp`` – element-wise exponential."""
    return wrap_surface_value(
        _pto.VexpOp(
            unwrap_surface_value(inp).type,
            unwrap_surface_value(inp),
            unwrap_surface_value(mask),
        ).result
    )


def vcgmax(v, mask):
    """``pto.vcgmax`` – group maximum reduction, surfaced as the lowest-lane scalar."""
    reduced = _pto.VcgmaxOp(
        unwrap_surface_value(v).type,
        unwrap_surface_value(v),
        unwrap_surface_value(mask),
    ).result
    return _extract_lowest_lane_scalar(reduced, mask)


def vcgadd(v, mask):
    """``pto.vcgadd`` – group sum reduction, surfaced as the lowest-lane scalar."""
    reduced = _pto.VcgaddOp(
        unwrap_surface_value(v).type,
        unwrap_surface_value(v),
        unwrap_surface_value(mask),
    ).result
    return _extract_lowest_lane_scalar(reduced, mask)


def vsubs(inp, scalar, mask):
    """``pto.vsubs`` – vector minus scalar under mask."""
    raw_scalar = _coerce_scalar_like_vector_element(inp, scalar, context="vsubs")
    neg_scalar = _negate_runtime_scalar(raw_scalar)
    return wrap_surface_value(
        _pto.VaddsOp(
            unwrap_surface_value(inp).type,
            unwrap_surface_value(inp),
            neg_scalar,
            unwrap_surface_value(mask),
        ).result
    )


# ── Tile-domain operations ────────────────────────────────────────────────────

def make_tensor_view(ptr, *, shape=None, strides=None):
    """
    ``pto.make_tensor_view`` – wrap a pointer as a tensor view.

    Type is inferred: rank from ``len(shape)``, element type from ``ptr``.
    """
    authored_ptr = ptr
    if shape is None:
        shape = getattr(authored_ptr, "shape", None)
    if strides is None:
        strides = getattr(authored_ptr, "strides", None)
    if shape is None or strides is None:
        raise TypeError("make_tensor_view() requires shape= and strides=, or a host tensor proxy carrying both")
    ptr = resolve_tensor_data_entry(authored_ptr)
    rank = len(shape)
    raw_ptr = unwrap_surface_value(ptr)
    elem = _pto.PtrType(raw_ptr.type).element_type
    tv_type = tensor_view_type(rank, elem)
    value = _pto.MakeTensorViewOp(
        tv_type,
        raw_ptr,
        _unwrap_sequence(shape),
        _unwrap_sequence(strides),
    ).result
    return TensorViewValue(value, shape=tuple(shape), strides=tuple(strides))


def _normalize_static_tile_shape(shape):
    static_shape = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError(
                "alloc_tile(shape=...) currently requires a static physical tile shape. "
                "Use constexpr/static integers for shape and place runtime metadata in valid_shape."
            )
        static_shape.append(dim)
    return tuple(static_shape)


def _split_valid_shape(shape, valid_shape):
    rank = len(shape)
    if valid_shape is None:
        return tuple(shape), None, None, tuple(shape)

    if len(valid_shape) != rank:
        raise TypeError(
            f"alloc_tile(valid_shape=...) rank mismatch: expected {rank} dims, got {len(valid_shape)}"
        )

    type_valid_shape = []
    surface_valid_shape = []
    valid_row = None
    valid_col = None
    for index, dim in enumerate(valid_shape):
        surface_valid_shape.append(dim)
        if isinstance(dim, bool):
            raise TypeError("alloc_tile(valid_shape=...) does not accept bool dimensions")
        if isinstance(dim, int):
            type_valid_shape.append(dim)
            continue
        type_valid_shape.append(-1)
        if index == 0:
            valid_row = dim
            continue
        if index == 1:
            valid_col = dim
            continue
        raise TypeError(
            "alloc_tile(valid_shape=...) currently only supports dynamic runtime metadata "
            "for the first two dimensions"
        )
    return tuple(type_valid_shape), valid_row, valid_col, tuple(surface_valid_shape)


def _uses_row_major_none_box_layout(blayout, slayout) -> bool:
    return str(blayout).lower() == "rowmajor" and str(slayout).lower() == "nonebox"


def _validate_authored_tile_row_alignment(shape, dtype, *, blayout, slayout):
    if not _uses_row_major_none_box_layout(blayout, slayout):
        return
    if not shape:
        return
    elem_bytewidth = _element_bytewidth(_resolve(dtype))
    row_bytes = shape[-1] * elem_bytewidth
    required_alignment = 32
    if row_bytes % required_alignment == 0:
        return
    raise tile_row_alignment_error(
        shape=shape,
        dtype=str(_resolve(dtype)),
        row_bytes=row_bytes,
        required_alignment=required_alignment,
    )


def partition_view(tv, *, offsets, sizes):
    """
    ``pto.partition_view`` – slice a tensor view.

    Type is inferred from the source tensor-view type.
    """
    spec = compose_partition_spec(tv, offsets=offsets, sizes=sizes)
    if spec is not None:
        source = spec.root_tensor_view
        offsets = spec.offsets
        sizes = spec.sizes
    else:
        source = tv

    raw_source = unwrap_surface_value(source)
    src_type = _pto.TensorViewType(raw_source.type)
    rank = src_type.rank
    elem = src_type.element_type
    ptv_type = part_tensor_view_type(rank, elem)
    value = _pto.PartitionViewOp(
        ptv_type,
        raw_source,
        _unwrap_sequence(offsets),
        _unwrap_sequence(sizes),
    ).result
    return wrap_surface_value(
        value,
        root_tensor_view=source if spec is None else spec.root_tensor_view,
        offsets=tuple(offsets),
        sizes=tuple(sizes),
    )


def alloc_tile(
    tile_type=None,
    *,
    shape=None,
    dtype=None,
    memory_space="ub",
    valid_shape=None,
    blayout: str = "RowMajor",
    slayout: str = "NoneBox",
    fractal_size: int = 512,
    pad: str = "Null",
    addr=None,
    valid_row=None,
    valid_col=None,
):
    """
    ``pto.alloc_tile``.

    Accepts either the authored surface form:

    ``alloc_tile(shape=[...], dtype=..., memory_space=...)``

    or the low-level explicit-type form:

    ``alloc_tile(tile_type, addr=..., valid_row=..., valid_col=...)``.
    """
    if tile_type is not None and shape is not None:
        raise TypeError("alloc_tile() accepts either tile_type or shape=/dtype=, not both")

    if tile_type is None:
        if shape is None or dtype is None:
            raise TypeError("alloc_tile() requires either tile_type or both shape= and dtype=")
        if addr is not None or valid_row is not None or valid_col is not None:
            raise TypeError(
                "alloc_tile(shape=..., dtype=...) uses the authored surface form; "
                "addr=/valid_row=/valid_col= are only supported with an explicit tile_type"
            )
        shape = _normalize_static_tile_shape(shape)
        _validate_authored_tile_row_alignment(shape, dtype, blayout=blayout, slayout=slayout)
        type_valid_shape, valid_row, valid_col, surface_valid_shape = _split_valid_shape(shape, valid_shape)
        from ._types import tile_buf_type
        tile_type = tile_buf_type(
            shape,
            dtype,
            type_valid_shape,
            blayout=blayout,
            address_space=memory_space,
            slayout=slayout,
            fractal_size=fractal_size,
            pad=pad,
        )
    else:
        surface_valid_shape = None

    value = _pto.AllocTileOp(
        _resolve(tile_type),
        addr=unwrap_surface_value(addr) if addr is not None else None,
        valid_row=unwrap_surface_value(valid_row) if valid_row is not None else None,
        valid_col=unwrap_surface_value(valid_col) if valid_col is not None else None,
    ).result
    if tile_type is not None and (valid_row is not None or valid_col is not None):
        parsed_tile_type = parse_tile_type_metadata(_resolve(tile_type))
        rank = len(shape) if shape is not None else len(parsed_tile_type["shape_dims"])
        surface_valid_shape = [None] * rank
        if rank >= 1:
            surface_valid_shape[0] = valid_row
        if rank >= 2:
            surface_valid_shape[1] = valid_col
        surface_valid_shape = tuple(surface_valid_shape)
    return wrap_surface_value(
        value,
        tile_metadata={
            "shape": shape,
            "dtype": dtype,
            "memory_space": memory_space,
            "valid_shape": surface_valid_shape,
        },
    )


def set_tile_valid_shape(tile, valid_shape):
    """Update the runtime valid-shape metadata of a rank-2 dynamic tile."""
    if len(valid_shape) != 2:
        raise TypeError(
            "tile.valid_shape assignment currently expects exactly two dimensions"
        )

    parsed_tile_type = parse_tile_type_metadata(unwrap_surface_value(tile).type)
    if parsed_tile_type is None:
        raise TypeError("tile.valid_shape assignment expects a tile_buf-backed value")
    if len(parsed_tile_type["shape_dims"]) != 2:
        raise TypeError("tile.valid_shape assignment currently only supports rank-2 tiles")
    if parsed_tile_type["valid_dims"] != (None, None):
        raise TypeError(
            "tile.valid_shape assignment requires a tile allocated with fully dynamic "
            "valid_shape=[..., ...]"
        )

    valid_row, valid_col = _unwrap_sequence(valid_shape)
    _pto.SetValidShapeOp(
        unwrap_surface_value(tile),
        valid_row,
        valid_col,
    )


def tload(part, tile):
    """``pto.tload ins(part) outs(tile)``."""
    _pto.TLoadOp(None, unwrap_surface_value(part), unwrap_surface_value(tile))


def tstore(tile, part):
    """``pto.tstore ins(tile) outs(part)``."""
    _pto.TStoreOp(None, unwrap_surface_value(tile), unwrap_surface_value(part))


def tmov(src, dst):
    """``pto.tmov ins(src) outs(dst)`` – move data between tile domains."""
    _pto.TMovOp(None, unwrap_surface_value(src), unwrap_surface_value(dst))


def as_ptr(value, result_ptr_type=None):
    """Materialize a typed pointer from a tile or tensor-view descriptor."""
    wrapped = wrap_surface_value(value)
    return emit_as_ptr(wrapped, result_ptr_type)


def _constant_like(value, mlir_type):
    value = unwrap_surface_value(value)
    if hasattr(value, "type"):
        return value
    if isinstance(value, float):
        return arith.ConstantOp(mlir_type, FloatAttr.get(mlir_type, value)).result
    return arith.ConstantOp(mlir_type, value).result


def _index_zero():
    return arith.ConstantOp(IndexType.get(), 0).result


def _infer_vreg_type_from_tile_slice(tile_slice: TileSliceValue):
    memref_type = MemRefType(tile_slice.type)
    elem_type = memref_type.element_type
    lanes = _elements_per_vreg(elem_type)
    return _resolve(vreg_type(lanes, elem_type))


def _elements_per_vreg(elem_type):
    if F32Type.isinstance(elem_type):
        bytewidth = 4
    elif any(cls.isinstance(elem_type) for cls in (F16Type, BF16Type)):
        bytewidth = 2
    elif IntegerType.isinstance(elem_type):
        width = IntegerType(elem_type).width
        if width % 8 != 0:
            raise TypeError(f"vlds/vsts tile-slice sugar does not support sub-byte integer element type {elem_type}")
        bytewidth = width // 8
    else:
        raise TypeError(f"vlds/vsts tile-slice sugar does not support element type {elem_type}")
    return 256 // bytewidth


def _infer_vreg_metadata(vector_value):
    raw_type = unwrap_surface_value(vector_value).type
    try:
        vreg_type = _pto.VRegType(raw_type)
        return vreg_type.lanes, vreg_type.element_type
    except Exception:
        text = str(raw_type)
        if not text.startswith("!pto.vreg<") or "x" not in text:
            raise TypeError(f"expected PTO vector-register type, got {raw_type}")
        body = text[len("!pto.vreg<"):-1]
        lanes_text, elem_text = body.split("x", 1)
        return int(lanes_text), Type.parse(elem_text)


def _extract_lowest_lane_scalar(vector_value, mask):
    lanes, elem_type = _infer_vreg_metadata(vector_value)
    tmp_tile = alloc_tile(shape=[1, lanes], dtype=elem_type, valid_shape=[1, 1])
    vsts_1pt(vector_value, tmp_tile.as_ptr(), _index_zero(), mask)
    from . import scalar as _scalar
    return _scalar.load(tmp_tile[0, 0])


def _element_bytewidth(elem_type):
    if F32Type.isinstance(elem_type):
        return 4
    if any(cls.isinstance(elem_type) for cls in (F16Type, BF16Type)):
        return 2
    if IntegerType.isinstance(elem_type):
        width = IntegerType(elem_type).width
        if width % 8 != 0:
            raise TypeError(f"unsupported sub-byte integer element type {elem_type}")
        return width // 8
    raise TypeError(f"unsupported element type {elem_type}")


def _mask_bits_for_dtype(dtype):
    elem_type = _resolve(dtype)
    bytewidth = _element_bytewidth(elem_type)
    if bytewidth == 4:
        return 32
    if bytewidth == 2:
        return 16
    if bytewidth == 1:
        return 8
    raise TypeError(f"make_mask(...) does not support dtype {elem_type}")


def _pset_op_for_mask_bits(mask_bits: int):
    return {
        8: _pto.PsetB8Op,
        16: _pto.PsetB16Op,
        32: _pto.PsetB32Op,
    }[mask_bits]


def _plt_op_for_mask_bits(mask_bits: int):
    return {
        8: _pto.PltB8Op,
        16: _pto.PltB16Op,
        32: _pto.PltB32Op,
    }[mask_bits]


def _coerce_i32(value, *, context: str):
    raw_value = unwrap_surface_value(value)
    i32_type = IntegerType.get_signless(32)
    if isinstance(raw_value, bool):
        raise TypeError(f"{context} does not accept bool values")
    if isinstance(raw_value, int):
        return arith.ConstantOp(i32_type, raw_value).result
    kind = classify_runtime_scalar_type(raw_value.type)
    if kind == "float":
        raise TypeError(f"{context} expects an integer-like scalar, got {raw_value.type}")
    if kind == "index":
        return arith.IndexCastOp(i32_type, raw_value).result
    if raw_value.type == i32_type:
        return raw_value
    width = IntegerType(raw_value.type).width
    if width < 32:
        return arith.ExtSIOp(i32_type, raw_value).result
    if width > 32:
        return arith.TruncIOp(i32_type, raw_value).result
    return raw_value


def _coerce_i64(value, *, context: str):
    raw_value = unwrap_surface_value(value)
    i64_type = IntegerType.get_signless(64)
    if isinstance(raw_value, bool):
        raise TypeError(f"{context} does not accept bool values")
    if isinstance(raw_value, int):
        return arith.ConstantOp(i64_type, raw_value).result
    kind = classify_runtime_scalar_type(raw_value.type)
    if kind == "float":
        raise TypeError(f"{context} expects an integer-like scalar, got {raw_value.type}")
    if kind == "index":
        return arith.IndexCastOp(i64_type, raw_value).result
    if raw_value.type == i64_type:
        return raw_value
    width = IntegerType(raw_value.type).width
    if width < 64:
        return arith.ExtSIOp(i64_type, raw_value).result
    if width > 64:
        return arith.TruncIOp(i64_type, raw_value).result
    return raw_value


def _i64_zero():
    return arith.ConstantOp(IntegerType.get_signless(64), 0).result


def _coerce_scalar_like_vector_element(vector_value, scalar_value, *, context: str):
    _, elem_type = _infer_vreg_metadata(vector_value)
    return coerce_scalar_to_type(scalar_value, elem_type, context=f"{context}(...)")


def _negate_runtime_scalar(value):
    raw_value = unwrap_surface_value(value)
    kind = classify_runtime_scalar_type(raw_value.type)
    zero = materialize_scalar_literal(0.0 if kind == "float" else 0, raw_value.type, context="_negate_runtime_scalar(...)")
    return emit_runtime_binary_op("sub", zero, raw_value)


def _mul_bytes(value, elem_type):
    factor = _element_bytewidth(_resolve(elem_type))
    raw_value = unwrap_surface_value(value)
    if isinstance(raw_value, int):
        return raw_value * factor
    return emit_runtime_binary_op("mul", raw_value, factor)


def _membar_attr(kind: str):
    normalized = str(kind)
    supported = {
        "VV_ALL",
        "VST_VLD",
        "VLD_VST",
        "VST_VST",
        "VS_ALL",
        "VST_LD",
        "VLD_ST",
        "VST_ST",
        "SV_ALL",
        "ST_VLD",
        "LD_VST",
        "ST_VST",
        "SS_ALL",
        "ST_LD",
        "LD_ST",
        "ST_ST",
    }
    if normalized not in supported:
        raise ValueError(f"unsupported mem_bar kind {kind!r}")
    return Attribute.parse(f"#pto.membar<{normalized}>")


def _acc_store_ub_dst_mode_attr(mode):
    normalized = {
        0: "single",
        1: "split_m",
        2: "split_n",
        "single": "single",
        "split_m": "split_m",
        "split_n": "split_n",
    }.get(mode if isinstance(mode, int) else str(mode).lower())
    if normalized is None:
        raise ValueError(f"unsupported mte_l0c_ub dst_mode {mode!r}")
    return Attribute.parse(f"#pto<acc_store_ub_dst_mode {normalized}>")


def _infer_dma_partition_row_stride(partition: PartitionTensorViewValue):
    if partition.shape is None or partition.strides is None:
        raise TypeError("mte_load/mte_store require partition view shape/stride metadata")
    outer_dims = list(partition.shape[:-1])
    non_unit = [i for i, dim in enumerate(outer_dims) if dim != 1]
    if len(non_unit) > 1:
        raise TypeError(
            "mte_load/mte_store currently only support partitions with at most one non-unit "
            "dimension before the contiguous innermost dimension"
        )
    if not non_unit:
        return 1, 0
    dim_index = non_unit[0]
    return partition.shape[dim_index], partition.strides[dim_index]


def _infer_dma_tile_geometry(tile: TileValue):
    if tile.shape is None:
        raise TypeError("mte_load/mte_store require tile shape metadata")
    if len(tile.shape) == 1:
        valid_cols = tile.valid_shape[0]
        return 1, valid_cols, tile.shape[0]
    if len(tile.shape) == 2:
        return tile.valid_shape[0], tile.valid_shape[1], tile.shape[1]
    raise TypeError("mte_load/mte_store currently only support rank-1 or rank-2 tiles")


def _infer_dma_2d_copy_signature(partition, tile, *, direction: str):
    row_count, src_row_stride = _infer_dma_partition_row_stride(partition)
    tile_rows, valid_cols, physical_cols = _infer_dma_tile_geometry(tile)
    if direction == "gm_to_ub":
        return row_count, valid_cols, _mul_bytes(src_row_stride, infer_tile_element_type(tile)), physical_cols * _element_bytewidth(infer_tile_element_type(tile))
    return row_count, valid_cols, physical_cols * _element_bytewidth(infer_tile_element_type(tile)), _mul_bytes(src_row_stride, infer_tile_element_type(tile))


def fill_tile(tile, value):
    """Broadcast a scalar into an entire tile."""
    wrapped_tile = wrap_surface_value(tile)
    scalar_value = _constant_like(value, infer_tile_element_type(wrapped_tile))
    _pto.TExpandsOp(scalar_value, unwrap_surface_value(wrapped_tile))


def make_mask(dtype, value):
    """Create a predicate mask matching *dtype* granularity."""
    mask_bits = _mask_bits_for_dtype(dtype)
    result_type = _resolve(mask_type(f"b{mask_bits}"))

    if isinstance(value, str):
        return wrap_surface_value(_pset_op_for_mask_bits(mask_bits)(result_type, value).result)

    raw_value = unwrap_surface_value(value)
    raw_value = _coerce_i32(raw_value, context="make_mask(..., value)")
    plt_op = _plt_op_for_mask_bits(mask_bits)(result_type, IntegerType.get_signless(32), raw_value)
    return MaskResultValue(plt_op.mask, plt_op.scalar_out)


# ── Hardware / sync ───────────────────────────────────────────────────────────

def mte_load(source, destination):
    """
    Convenience GM->on-chip load surface.

    Current scope is intentionally narrow: contiguous rank-1 or squeezed-rank-2
    partition views lowering into VEC or MAT tiles.
    """
    source = wrap_surface_value(source)
    destination = wrap_surface_value(destination)
    if not isinstance(source, PartitionTensorViewValue) or not isinstance(destination, TileValue):
        raise TypeError("mte_load(source, destination) expects (PartitionTensorView, Tile)")

    src_ptr = emit_as_ptr(source)
    dst_ptr = emit_as_ptr(destination)
    row_count, valid_cols, src_row_stride, dst_row_stride = _infer_dma_2d_copy_signature(
        source, destination, direction="gm_to_ub"
    )
    destination_type = parse_tile_type_metadata(unwrap_surface_value(destination).type)
    if destination_type is None:
        raise TypeError("mte_load(source, destination) expects a tile_buf-backed destination")
    destination_space = destination_type["memory_space"]
    len_burst = _coerce_i64(_mul_bytes(valid_cols, infer_tile_element_type(destination)), context="mte_load len_burst")
    n_burst = _coerce_i64(row_count, context="mte_load n_burst")
    src_stride = _coerce_i64(src_row_stride, context="mte_load src_stride")
    dst_stride = _coerce_i64(dst_row_stride, context="mte_load dst_stride")

    if destination_space == "vec":
        _pto.MteGmUbOp(
            unwrap_surface_value(src_ptr),
            unwrap_surface_value(dst_ptr),
            _i64_zero(),
            len_burst,
            n_burst,
            src_stride,
            dst_stride,
            [],
            [],
            [],
        )
        return

    if destination_space == "mat":
        _pto.MteGmL1Op(
            unwrap_surface_value(src_ptr),
            unwrap_surface_value(dst_ptr),
            len_burst,
            n_burst,
            src_stride,
            dst_stride,
            [],
            [],
            [],
        )
        return

    raise TypeError(
        "mte_load(source, destination) currently supports VEC or MAT tile destinations, "
        f"got memory_space={destination_space!r}"
    )


def mte_store(source, destination):
    """Convenience UB->GM store surface matching ``mte_load`` scope."""
    source = wrap_surface_value(source)
    destination = wrap_surface_value(destination)
    if not isinstance(source, TileValue) or not isinstance(destination, PartitionTensorViewValue):
        raise TypeError("mte_store(source, destination) expects (Tile, PartitionTensorView)")

    src_ptr = emit_as_ptr(source)
    dst_ptr = emit_as_ptr(destination)
    row_count, valid_cols, src_row_stride, dst_row_stride = _infer_dma_2d_copy_signature(
        destination, source, direction="ub_to_gm"
    )
    _pto.MteUbGmOp(
        unwrap_surface_value(src_ptr),
        unwrap_surface_value(dst_ptr),
        _coerce_i64(_mul_bytes(valid_cols, infer_tile_element_type(source)), context="mte_store len_burst"),
        _coerce_i64(row_count, context="mte_store n_burst"),
        _coerce_i64(src_row_stride, context="mte_store src_stride"),
        _coerce_i64(dst_row_stride, context="mte_store dst_stride"),
        [],
        [],
        [],
    )


def mem_bar(barrier_type):
    """``pto.mem_bar`` with a small authored enum surface."""
    barrier_name = getattr(barrier_type, "value", barrier_type)
    _pto.MemBarOp(kind=_membar_attr(barrier_name))


def mte_l1_l0a(source, destination, m, k, *, transpose=False):
    """``pto.mte_l1_l0a`` – cube-side LEFT staging."""
    _pto.MteL1L0aOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(m, context="mte_l1_l0a m"),
        _coerce_i64(k, context="mte_l1_l0a k"),
        transpose=transpose,
    )


def mte_l1_l0b(source, destination, k, n, *, transpose=False):
    """``pto.mte_l1_l0b`` – cube-side RIGHT staging."""
    _pto.MteL1L0bOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(k, context="mte_l1_l0b k"),
        _coerce_i64(n, context="mte_l1_l0b n"),
        transpose=transpose,
    )


def mte_l0c_ub(source, destination, m, n, src_stride, dst_stride, sub_blockid=0, *, dst_mode="single"):
    """``pto.mte_l0c_ub`` – ACC to UB store."""
    _pto.MteL0cUbOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(m, context="mte_l0c_ub m"),
        _coerce_i64(n, context="mte_l0c_ub n"),
        _coerce_i64(src_stride, context="mte_l0c_ub src_stride"),
        _coerce_i64(dst_stride, context="mte_l0c_ub dst_stride"),
        _acc_store_ub_dst_mode_attr(dst_mode),
        sub_blockid=_coerce_i64(sub_blockid, context="mte_l0c_ub sub_blockid"),
    )


def mad(lhs, rhs, dst, m, n, k):
    """``pto.mad`` – cube matmul accumulate."""
    _pto.MadOp(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
        _coerce_i64(m, context="mad m"),
        _coerce_i64(n, context="mad n"),
        _coerce_i64(k, context="mad k"),
    )

def get_block_idx():
    """``pto.get_block_idx`` → i64 block index."""
    return wrap_surface_value(_pto.GetBlockIdxOp().result)


def get_block_num():
    """``pto.get_block_num`` → i64 block count."""
    return wrap_surface_value(_pto.GetBlockNumOp().result)


def get_subblock_idx():
    """``pto.get_subblock_idx`` → i64 subblock index."""
    return wrap_surface_value(_pto.GetSubBlockIdxOp().result)


def get_subblock_num():
    """``pto.get_subblock_num`` → i64 subblock count."""
    return wrap_surface_value(_pto.GetSubBlockNumOp().result)


def store_vfsimt_info(dim_z, dim_y, dim_x):
    """``pto.store_vfsimt_info`` – configure the SIMT VF launch descriptor."""
    _pto.StoreVfSimtInfoOp(
        unwrap_surface_value(dim_z),
        unwrap_surface_value(dim_y),
        unwrap_surface_value(dim_x),
    )


def get_tid_x():
    """``pto.get_tid_x`` → i32 SIMT lane X coordinate."""
    return wrap_surface_value(_pto.GetTidXOp().result)


def get_tid_y():
    """``pto.get_tid_y`` → i32 SIMT lane Y coordinate."""
    return wrap_surface_value(_pto.GetTidYOp().result)


def get_tid_z():
    """``pto.get_tid_z`` → i32 SIMT lane Z coordinate."""
    return wrap_surface_value(_pto.GetTidZOp().result)


def pipe_barrier(pipe):
    """``pto.pipe_barrier(pipe)`` – drain the specified hardware pipeline."""
    _pto.BarrierOp(_pipe_attr(pipe))


def set_flag(src: str, dst: str, *, event_id: int = 0):
    """``pto.set_flag[src, dst, event_id]``.

    Accepts short pipe names (``"MTE2"``, ``"V"``, …) or full ``"PIPE_MTE2"``
    names.  ``event_id`` is an integer in ``[0, 7]``.
    """
    _pto.set_flag(_pipe_attr(src), _pipe_attr(dst), _event_attr(event_id))


def wait_flag(src: str, dst: str, *, event_id: int = 0):
    """``pto.wait_flag[src, dst, event_id]``."""
    _pto.wait_flag(_pipe_attr(src), _pipe_attr(dst), _event_attr(event_id))


__all__ = [
    "const",
    "castptr", "addptr",
    "vlds", "vbrc_load", "vsts", "vsts_1pt",
    "plt_b32", "pset_b32", "make_mask",
    "vadd", "vmul", "vmax", "vdiv",
    "vcmax", "vcadd", "vdup", "vexpdif",
    "vexp", "vcgmax", "vcgadd", "vsubs",
    "make_tensor_view", "partition_view",
    "alloc_tile", "tload", "tstore", "tmov", "as_ptr",
    "mte_load", "mte_store", "mem_bar",
    "mte_l1_l0a", "mte_l1_l0b", "mte_l0c_ub", "mad",
    "get_block_idx", "get_block_num", "get_subblock_idx", "get_subblock_num",
    "store_vfsimt_info", "get_tid_x", "get_tid_y", "get_tid_z",
    "pipe_barrier", "set_flag", "wait_flag",
]
