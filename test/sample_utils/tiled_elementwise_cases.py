from mlir.ir import (
    Context,
    F32Type,
    IndexType,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
)
from mlir.dialects import arith, func, pto

TILE_ROWS = 32
TILE_COLS = 32
SCALAR_VALUE = 3.14


def _idx_const(ctx, value: int):
    return arith.ConstantOp(IndexType.get(ctx), value).result


def _build_vec_tile_type(
    ctx, *, rows: int, cols: int, dynamic_valid: bool, valid_rows: int, valid_cols: int
):
    f32 = F32Type.get(ctx)
    vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
    bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
    sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
    pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
    cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
    valid_shape = [-1, -1] if dynamic_valid else [valid_rows, valid_cols]
    return pto.TileBufType.get([rows, cols], f32, vec, valid_shape, cfg, ctx)


def _tile_view_type(ctx, *, rows: int, cols: int):
    f32 = F32Type.get(ctx)
    return pto.PartitionTensorViewType.get([rows, cols], f32, ctx)


def _build_case(
    *, kernel_name: str, op_name: str, rows: int, cols: int, dynamic_shape: bool
):
    if rows not in (1, TILE_ROWS):
        raise ValueError(f"unsupported rows={rows}, expected 1 or {TILE_ROWS}")
    if cols not in (TILE_COLS, TILE_COLS * 3):
        raise ValueError(
            f"unsupported cols={cols}, expected {TILE_COLS} or {TILE_COLS * 3}"
        )

    is_scalar = op_name in {"muls", "divs"}
    if op_name not in {"mul", "muls", "div", "divs"}:
        raise ValueError(f"unsupported op_name={op_name}")

    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view = _tile_view_type(ctx, rows=rows, cols=cols)
            tile_type = _build_vec_tile_type(
                ctx,
                rows=rows,
                cols=cols,
                dynamic_valid=dynamic_shape,
                valid_rows=rows,
                valid_cols=cols,
            )

            fn_inputs = [ptr_f32, ptr_f32] if is_scalar else [ptr_f32, ptr_f32, ptr_f32]
            if dynamic_shape:
                i32 = IntegerType.get_signless(32, ctx)
                fn_inputs.extend([i32, i32])
            fn_ty = func.FunctionType.get(fn_inputs, [])

            with InsertionPoint(module.body):
                fn = func.FuncOp(kernel_name, fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = _idx_const(ctx, 0)
                c1 = _idx_const(ctx, 1)
                c_rows = _idx_const(ctx, rows)
                c_cols = _idx_const(ctx, cols)

                if dynamic_shape:
                    if is_scalar:
                        src_ptr, dst_ptr, row_arg, col_arg = entry.arguments
                    else:
                        lhs_ptr, rhs_ptr, dst_ptr, row_arg, col_arg = entry.arguments
                    shape_rows = arith.IndexCastOp(IndexType.get(ctx), row_arg).result
                    shape_cols = arith.IndexCastOp(IndexType.get(ctx), col_arg).result
                else:
                    if is_scalar:
                        src_ptr, dst_ptr = entry.arguments
                    else:
                        lhs_ptr, rhs_ptr, dst_ptr = entry.arguments
                    shape_rows = c_rows
                    shape_cols = c_cols

                if is_scalar:
                    src_tv = pto.MakeTensorViewOp(
                        tv2_f32, src_ptr, [shape_rows, shape_cols], [shape_cols, c1]
                    ).result
                    dst_tv = pto.MakeTensorViewOp(
                        tv2_f32, dst_ptr, [shape_rows, shape_cols], [shape_cols, c1]
                    ).result
                else:
                    lhs_tv = pto.MakeTensorViewOp(
                        tv2_f32, lhs_ptr, [shape_rows, shape_cols], [shape_cols, c1]
                    ).result
                    rhs_tv = pto.MakeTensorViewOp(
                        tv2_f32, rhs_ptr, [shape_rows, shape_cols], [shape_cols, c1]
                    ).result
                    dst_tv = pto.MakeTensorViewOp(
                        tv2_f32, dst_ptr, [shape_rows, shape_cols], [shape_cols, c1]
                    ).result

                if is_scalar:
                    src_view = pto.PartitionViewOp(
                        tile_view,
                        src_tv,
                        offsets=[c0, c0],
                        sizes=[shape_rows, shape_cols],
                    ).result
                else:
                    lhs_view = pto.PartitionViewOp(
                        tile_view,
                        lhs_tv,
                        offsets=[c0, c0],
                        sizes=[shape_rows, shape_cols],
                    ).result
                    rhs_view = pto.PartitionViewOp(
                        tile_view,
                        rhs_tv,
                        offsets=[c0, c0],
                        sizes=[shape_rows, shape_cols],
                    ).result
                dst_view = pto.PartitionViewOp(
                    tile_view,
                    dst_tv,
                    offsets=[c0, c0],
                    sizes=[shape_rows, shape_cols],
                ).result

                alloc_kwargs = {}
                if dynamic_shape:
                    alloc_kwargs = {"valid_row": shape_rows, "valid_col": shape_cols}

                src_tile = pto.AllocTileOp(tile_type, **alloc_kwargs).result
                dst_tile = pto.AllocTileOp(tile_type, **alloc_kwargs).result

                if is_scalar:
                    scalar = arith.ConstantOp(f32, SCALAR_VALUE).result
                    pto.TLoadOp(None, src_view, src_tile)
                    if op_name == "muls":
                        pto.TMulSOp(src_tile, scalar, dst_tile)
                    else:
                        pto.TDivSOp(src_tile, scalar, dst_tile)
                else:
                    rhs_tile = pto.AllocTileOp(tile_type, **alloc_kwargs).result
                    pto.TLoadOp(None, lhs_view, src_tile)
                    pto.TLoadOp(None, rhs_view, rhs_tile)
                    if op_name == "mul":
                        pto.TMulOp(src_tile, rhs_tile, dst_tile)
                    else:
                        pto.TDivOp(src_tile, rhs_tile, dst_tile)

                pto.TStoreOp(None, dst_tile, dst_view)

                func.ReturnOp([])

            module.operation.verify()
            return module


def build_binary_case(
    *, kernel_name: str, op_name: str, rows: int, cols: int, dynamic_shape: bool
):
    return _build_case(
        kernel_name=kernel_name,
        op_name=op_name,
        rows=rows,
        cols=cols,
        dynamic_shape=dynamic_shape,
    )


def build_scalar_case(
    *, kernel_name: str, op_name: str, rows: int, cols: int, dynamic_shape: bool
):
    return _build_case(
        kernel_name=kernel_name,
        op_name=op_name,
        rows=rows,
        cols=cols,
        dynamic_shape=dynamic_shape,
    )
