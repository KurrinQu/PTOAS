from mlir.ir import (
    BF16Type,
    Context,
    F16Type,
    F32Type,
    IndexType,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    StringAttr,
    UnitAttr,
)
from mlir.dialects import arith, func, pto

_BINARY_OPS = {
    "tadd": pto.TAddOp,
    "tsub": pto.TSubOp,
    "tmul": pto.TMulOp,
    "tdiv": pto.TDivOp,
    "tmax": pto.TMaxOp,
    "tmin": pto.TMinOp,
}

_SCALAR_OPS = {
    "tadds": pto.TAddSOp,
    "tsubs": pto.TSubSOp,
    "tmuls": pto.TMulSOp,
    "tdivs": pto.TDivSOp,
    "tmaxs": pto.TMaxSOp,
    "tmins": pto.TMinSOp,
}

_PAD_VALUES = {
    "null": pto.PadValue.Null,
    "max": pto.PadValue.Max,
}


def _idx_const(ctx: Context, value: int):
    return arith.ConstantOp(IndexType.get(ctx), value).result


def _elem_type(ctx: Context, dtype_token: str):
    if dtype_token == "f32":
        return F32Type.get(ctx)
    if dtype_token == "f16":
        return F16Type.get(ctx)
    if dtype_token == "bf16":
        return BF16Type.get(ctx)
    if dtype_token == "i32":
        return IntegerType.get_signless(32, ctx)
    if dtype_token == "i16":
        return IntegerType.get_signless(16, ctx)
    if dtype_token == "i8":
        return IntegerType.get_signless(8, ctx)
    if dtype_token == "u32":
        return IntegerType.get_unsigned(32, ctx)
    if dtype_token == "u16":
        return IntegerType.get_unsigned(16, ctx)
    if dtype_token == "u8":
        return IntegerType.get_unsigned(8, ctx)
    raise ValueError(f"unsupported dtype_token={dtype_token}")


def _tile_buf_type(
    ctx: Context,
    *,
    shape: tuple[int, int],
    elem_ty,
    valid_shape: tuple[int, int],
    pad_value: str,
):
    vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
    bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
    sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
    pd = pto.PadValueAttr.get(_PAD_VALUES[pad_value], ctx)
    cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
    return pto.TileBufType.get(list(shape), elem_ty, vec, list(valid_shape), cfg, ctx)


def _tensor_view_type(ctx: Context, elem_ty):
    return pto.TensorViewType.get(2, elem_ty, ctx)


def _partition_view_type(ctx: Context, *, shape: tuple[int, int], elem_ty):
    return pto.PartitionTensorViewType.get(list(shape), elem_ty, ctx)


def _make_view(ctx: Context, ptr, elem_ty, *, shape: tuple[int, int]):
    tv_ty = _tensor_view_type(ctx, elem_ty)
    pv_ty = _partition_view_type(ctx, shape=shape, elem_ty=elem_ty)
    c0 = _idx_const(ctx, 0)
    c1 = _idx_const(ctx, 1)
    c_rows = _idx_const(ctx, shape[0])
    c_cols = _idx_const(ctx, shape[1])
    tv = pto.MakeTensorViewOp(tv_ty, ptr, [c_rows, c_cols], [c_cols, c1]).result
    pv = pto.PartitionViewOp(pv_ty, tv, offsets=[c0, c0], sizes=[c_rows, c_cols]).result
    return pv


def _subset_if_needed(
    tile, *, src_shape: tuple[int, int], dst_shape: tuple[int, int], ctx: Context
):
    if src_shape == dst_shape:
        return tile
    c0 = _idx_const(ctx, 0)
    return pto.SubsetOp(tile, [c0, c0], sizes=[dst_shape[0], dst_shape[1]]).result


def _clamp_valid_shape(
    valid_shape: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, int]:
    return (
        min(valid_shape[0], shape[0]),
        min(valid_shape[1], shape[1]),
    )

def _target_arch_for_binary_case(op_name: str, dtype_token: str) -> str | None:
    # Integer binary tdiv is only accepted on the A5 verifier path.
    if op_name == "tdiv" and dtype_token in {"i16", "i32", "u16", "u32"}:
        return "a5"
    return None


def _target_arch_for_scalar_case(op_name: str, dtype_token: str) -> str | None:
    a5_only_scalar_dtypes = {
        "tadds": {"i8", "u8", "bf16"},
        "tsubs": {"i8", "u8", "bf16"},
        "tmuls": {"i8", "u8", "bf16"},
        "tmaxs": {"i8", "u8", "bf16"},
        "tmins": {"i8", "u8", "bf16"},
        "tdivs": {"i8", "u8"},
    }
    if dtype_token in a5_only_scalar_dtypes.get(op_name, set()):
        return "a5"
    return None


def _needs_dynamic_source_tile(
    valid_shape: tuple[int, int], shape: tuple[int, int]
) -> bool:
    return _clamp_valid_shape(valid_shape, shape) != shape


def _dynamic_valid_shape() -> tuple[int, int]:
    return (-1, -1)


def _set_valid_shape(tile, *, valid_shape: tuple[int, int], ctx: Context):
    pto.SetValidShapeOp(
        tile, _idx_const(ctx, valid_shape[0]), _idx_const(ctx, valid_shape[1])
    )
    return tile


def build_binary_case(
    *,
    kernel_name: str,
    op_name: str,
    dtype_token: str,
    dst_shape: tuple[int, int],
    src0_shape: tuple[int, int],
    src1_shape: tuple[int, int],
    valid_shape: tuple[int, int],
):
    if op_name not in _BINARY_OPS:
        raise ValueError(f"unsupported binary op_name={op_name}")

    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()
            target_arch = _target_arch_for_binary_case(op_name, dtype_token)
            if target_arch is not None:
                module.operation.attributes["pto.target_arch"] = StringAttr.get(
                    target_arch, ctx
                )
            elem_ty = _elem_type(ctx, dtype_token)
            ptr_ty = pto.PtrType.get(elem_ty, ctx)

            with InsertionPoint(module.body):
                fn = func.FuncOp(
                    kernel_name, func.FunctionType.get([ptr_ty, ptr_ty, ptr_ty], [])
                )
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                src0_ptr, src1_ptr, dst_ptr = entry.arguments

                src0_view = _make_view(ctx, src0_ptr, elem_ty, shape=src0_shape)
                src1_view = _make_view(ctx, src1_ptr, elem_ty, shape=src1_shape)
                dst_view = _make_view(ctx, dst_ptr, elem_ty, shape=valid_shape)
                src0_dynamic_valid = _needs_dynamic_source_tile(valid_shape, src0_shape)
                src1_dynamic_valid = _needs_dynamic_source_tile(valid_shape, src1_shape)

                src0_tile = pto.AllocTileOp(
                    _tile_buf_type(
                        ctx,
                        shape=src0_shape,
                        elem_ty=elem_ty,
                        valid_shape=(
                            _dynamic_valid_shape()
                            if src0_dynamic_valid
                            else _clamp_valid_shape(valid_shape, src0_shape)
                        ),
                        pad_value="null",
                    ),
                    **(
                        {
                            "valid_row": _idx_const(ctx, src0_shape[0]),
                            "valid_col": _idx_const(ctx, src0_shape[1]),
                        }
                        if src0_dynamic_valid
                        else {}
                    ),
                ).result
                src1_tile = pto.AllocTileOp(
                    _tile_buf_type(
                        ctx,
                        shape=src1_shape,
                        elem_ty=elem_ty,
                        valid_shape=(
                            _dynamic_valid_shape()
                            if src1_dynamic_valid
                            else _clamp_valid_shape(valid_shape, src1_shape)
                        ),
                        pad_value="null",
                    ),
                    **(
                        {
                            "valid_row": _idx_const(ctx, src1_shape[0]),
                            "valid_col": _idx_const(ctx, src1_shape[1]),
                        }
                        if src1_dynamic_valid
                        else {}
                    ),
                ).result
                dst_tile = pto.AllocTileOp(
                    _tile_buf_type(
                        ctx,
                        shape=dst_shape,
                        elem_ty=elem_ty,
                        valid_shape=valid_shape,
                        pad_value="null",
                    )
                ).result

                pto.TLoadOp(None, src0_view, src0_tile)
                pto.TLoadOp(None, src1_view, src1_tile)
                if src0_dynamic_valid:
                    src0_tile = _set_valid_shape(
                        src0_tile,
                        valid_shape=_clamp_valid_shape(valid_shape, src0_shape),
                        ctx=ctx,
                    )
                if src1_dynamic_valid:
                    src1_tile = _set_valid_shape(
                        src1_tile,
                        valid_shape=_clamp_valid_shape(valid_shape, src1_shape),
                        ctx=ctx,
                    )

                use_src0 = _subset_if_needed(
                    src0_tile,
                    src_shape=src0_shape,
                    dst_shape=dst_shape,
                    ctx=ctx,
                )
                use_src1 = _subset_if_needed(
                    src1_tile,
                    src_shape=src1_shape,
                    dst_shape=dst_shape,
                    ctx=ctx,
                )
                _BINARY_OPS[op_name](use_src0, use_src1, dst_tile)
                pto.TStoreOp(None, dst_tile, dst_view)

                func.ReturnOp([])

            module.operation.verify()
            return module


def build_scalar_case(
    *,
    kernel_name: str,
    op_name: str,
    dtype_token: str,
    dst_shape: tuple[int, int],
    src_shape: tuple[int, int],
    valid_shape: tuple[int, int],
    pad_value: str = "null",
):
    if op_name not in _SCALAR_OPS:
        raise ValueError(f"unsupported scalar op_name={op_name}")
    if pad_value not in _PAD_VALUES:
        raise ValueError(f"unsupported pad_value={pad_value}")

    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()
            target_arch = _target_arch_for_scalar_case(op_name, dtype_token)
            if target_arch is not None:
                module.operation.attributes["pto.target_arch"] = StringAttr.get(
                    target_arch, ctx
                )
            elem_ty = _elem_type(ctx, dtype_token)
            ptr_ty = pto.PtrType.get(elem_ty, ctx)

            with InsertionPoint(module.body):
                fn = func.FuncOp(
                    kernel_name, func.FunctionType.get([ptr_ty, ptr_ty, ptr_ty], [])
                )
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                src_ptr, scalar_ptr, dst_ptr = entry.arguments

                src_view = _make_view(ctx, src_ptr, elem_ty, shape=src_shape)
                dst_view = _make_view(ctx, dst_ptr, elem_ty, shape=valid_shape)
                src_dynamic_valid = _needs_dynamic_source_tile(valid_shape, src_shape)

                src_tile = pto.AllocTileOp(
                    _tile_buf_type(
                        ctx,
                        shape=src_shape,
                        elem_ty=elem_ty,
                        valid_shape=(
                            _dynamic_valid_shape()
                            if src_dynamic_valid
                            else _clamp_valid_shape(valid_shape, src_shape)
                        ),
                        pad_value=pad_value,
                    ),
                    **(
                        {
                            "valid_row": _idx_const(ctx, src_shape[0]),
                            "valid_col": _idx_const(ctx, src_shape[1]),
                        }
                        if src_dynamic_valid
                        else {}
                    ),
                ).result
                dst_tile = pto.AllocTileOp(
                    _tile_buf_type(
                        ctx,
                        shape=dst_shape,
                        elem_ty=elem_ty,
                        valid_shape=valid_shape,
                        pad_value=pad_value,
                    )
                ).result

                pto.TLoadOp(None, src_view, src_tile)
                if src_dynamic_valid:
                    src_tile = _set_valid_shape(
                        src_tile,
                        valid_shape=_clamp_valid_shape(valid_shape, src_shape),
                        ctx=ctx,
                    )
                scalar = pto.load_scalar(elem_ty, scalar_ptr, _idx_const(ctx, 0))

                work_src = _subset_if_needed(
                    src_tile,
                    src_shape=src_shape,
                    dst_shape=dst_shape,
                    ctx=ctx,
                )
                _SCALAR_OPS[op_name](work_src, scalar, dst_tile)
                pto.TStoreOp(None, dst_tile, dst_view)

                func.ReturnOp([])

            module.operation.verify()
            return module
