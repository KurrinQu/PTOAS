from mlir.ir import Context, F32Type, IndexType, InsertionPoint, Location, Module
from mlir.dialects import arith, func, pto

ROWS = 32
COLS = 96

MIXED_SCALE = 0.1767767
MIXED_NEG_CLAMP = -4.0001
MIXED_POS_CLAMP = 4.0001
MIXED_TAYLOR_C2 = 0.5
MIXED_TAYLOR_C3 = 0.1666667
MIXED_TAYLOR_C4 = 0.04166667
MIXED_BIAS = 1.0001
MIXED_DIVISOR = 96.0001

SCALAR_MUL = 1.25
SCALAR_ADD = -0.75
SCALAR_MAX = -1.5
SCALAR_MIN = 2.5
SCALAR_DIV = 0.5


def _idx_const(ctx, value: int):
    return arith.ConstantOp(IndexType.get(ctx), value).result


def _f32_const(ctx, value: float):
    return arith.ConstantOp(F32Type.get(ctx), float(value)).result


def _tile_view_type(ctx):
    f32 = F32Type.get(ctx)
    return pto.PartitionTensorViewType.get([ROWS, COLS], f32, ctx)


def _vec_tile_type(ctx):
    f32 = F32Type.get(ctx)
    vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
    bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
    sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
    pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
    cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
    return pto.TileBufType.get([ROWS, COLS], f32, vec, [ROWS, COLS], cfg, ctx)


def _alloc_tile(tile_type):
    return pto.AllocTileOp(tile_type).result


def _make_partition_view(ctx, ptr):
    f32 = F32Type.get(ctx)
    tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
    tile_view = _tile_view_type(ctx)
    c0 = _idx_const(ctx, 0)
    c1 = _idx_const(ctx, 1)
    c_rows = _idx_const(ctx, ROWS)
    c_cols = _idx_const(ctx, COLS)
    tv = pto.MakeTensorViewOp(tv2_f32, ptr, [c_rows, c_cols], [c_cols, c1]).result
    return pto.PartitionViewOp(
        tile_view, tv, offsets=[c0, c0], sizes=[c_rows, c_cols]
    ).result


def _build_module(kernel_name: str, ptr_count: int, body_builder):
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()
            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            fn_ty = func.FunctionType.get([ptr_f32] * ptr_count, [])

            with InsertionPoint(module.body):
                fn = func.FuncOp(kernel_name, fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                body_builder(ctx, entry.arguments)
                func.ReturnOp([])

            module.operation.verify()
            return module


def build_mixed_chain_case(*, kernel_name: str):
    def body(ctx, args):
        src_view = _make_partition_view(ctx, args[0])
        dst_view = _make_partition_view(ctx, args[1])
        tile_type = _vec_tile_type(ctx)

        src_tile = _alloc_tile(tile_type)
        out_tile = _alloc_tile(tile_type)
        tmp0 = _alloc_tile(tile_type)
        tmp1 = _alloc_tile(tile_type)
        tmp2 = _alloc_tile(tile_type)

        scale = _f32_const(ctx, MIXED_SCALE)
        neg_clamp = _f32_const(ctx, MIXED_NEG_CLAMP)
        pos_clamp = _f32_const(ctx, MIXED_POS_CLAMP)
        c2 = _f32_const(ctx, MIXED_TAYLOR_C2)
        c3 = _f32_const(ctx, MIXED_TAYLOR_C3)
        c4 = _f32_const(ctx, MIXED_TAYLOR_C4)
        bias = _f32_const(ctx, MIXED_BIAS)
        divisor = _f32_const(ctx, MIXED_DIVISOR)

        pto.TLoadOp(None, src_view, src_tile)
        pto.TMulSOp(src_tile, scale, src_tile)
        pto.TMaxSOp(src_tile, neg_clamp, src_tile)
        pto.TMinSOp(src_tile, pos_clamp, src_tile)

        pto.TMulOp(src_tile, src_tile, tmp0)
        pto.TMulOp(tmp0, src_tile, tmp1)
        pto.TMulOp(tmp1, src_tile, tmp2)

        pto.TMulSOp(tmp0, c2, tmp0)
        pto.TMulSOp(tmp1, c3, tmp1)
        pto.TMulSOp(tmp2, c4, tmp2)

        pto.TAddOp(src_tile, tmp0, out_tile)
        pto.TAddOp(out_tile, tmp1, out_tile)
        pto.TAddOp(out_tile, tmp2, out_tile)
        pto.TAddSOp(out_tile, bias, out_tile)
        pto.TDivSOp(out_tile, divisor, out_tile)
        pto.TStoreOp(None, out_tile, dst_view)

    return _build_module(kernel_name, 2, body)


def build_binary_chain_case(*, kernel_name: str):
    def body(ctx, args):
        lhs_view = _make_partition_view(ctx, args[0])
        rhs_view = _make_partition_view(ctx, args[1])
        dst_view = _make_partition_view(ctx, args[2])
        tile_type = _vec_tile_type(ctx)

        lhs_tile = _alloc_tile(tile_type)
        rhs_tile = _alloc_tile(tile_type)
        tmp0 = _alloc_tile(tile_type)
        tmp1 = _alloc_tile(tile_type)
        tmp2 = _alloc_tile(tile_type)
        tmp3 = _alloc_tile(tile_type)
        out_tile = _alloc_tile(tile_type)

        pto.TLoadOp(None, lhs_view, lhs_tile)
        pto.TLoadOp(None, rhs_view, rhs_tile)

        pto.TMulOp(lhs_tile, rhs_tile, tmp0)
        pto.TDivOp(tmp0, rhs_tile, tmp1)
        pto.TAddOp(tmp1, tmp0, tmp2)
        pto.TSubOp(tmp2, lhs_tile, tmp3)
        pto.TMaxOp(tmp3, rhs_tile, out_tile)
        pto.TMinOp(out_tile, tmp2, out_tile)
        pto.TStoreOp(None, out_tile, dst_view)

    return _build_module(kernel_name, 3, body)


def build_scalar_chain_case(*, kernel_name: str):
    def body(ctx, args):
        src_view = _make_partition_view(ctx, args[0])
        dst_view = _make_partition_view(ctx, args[1])
        tile_type = _vec_tile_type(ctx)

        src_tile = _alloc_tile(tile_type)
        tmp0 = _alloc_tile(tile_type)
        tmp1 = _alloc_tile(tile_type)
        tmp2 = _alloc_tile(tile_type)
        tmp3 = _alloc_tile(tile_type)
        out_tile = _alloc_tile(tile_type)

        mul_scalar = _f32_const(ctx, SCALAR_MUL)
        add_scalar = _f32_const(ctx, SCALAR_ADD)
        max_scalar = _f32_const(ctx, SCALAR_MAX)
        min_scalar = _f32_const(ctx, SCALAR_MIN)
        div_scalar = _f32_const(ctx, SCALAR_DIV)

        pto.TLoadOp(None, src_view, src_tile)
        pto.TMulSOp(src_tile, mul_scalar, tmp0)
        pto.TAddSOp(tmp0, add_scalar, tmp1)
        pto.TMaxSOp(tmp1, max_scalar, tmp2)
        pto.TMinSOp(tmp2, min_scalar, tmp3)
        pto.TDivSOp(tmp3, div_scalar, out_tile)
        pto.TStoreOp(None, out_tile, dst_view)

    return _build_module(kernel_name, 2, body)
