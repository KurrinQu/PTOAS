from mlir.ir import Context, InsertionPoint, Location, Module
from mlir.dialects import arith, func, pto
from mlir.ir import F32Type, IndexType, IntegerType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            f32 = F32Type.get(ctx)
            index = IndexType.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            u8 = IntegerType.get_unsigned(8, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            tile_buf_32 = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)
            tile_buf_u8 = pto.TileBufType.get([32, 32], u8, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_f32, ptr_f32], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("vector_arith_44_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(index, 0).result
                c1 = arith.ConstantOp(index, 1).result
                c32 = arith.ConstantOp(index, 32).result
                scale = arith.ConstantOp(f32, 2.0).result
                bias = arith.ConstantOp(f32, -0.125).result
                slope = arith.ConstantOp(f32, 0.3125).result

                arg_a, arg_b, arg_c, arg_out = entry.arguments

                tv_a = pto.MakeTensorViewOp(
                    tv2_f32, arg_a, [c32, c32], [c32, c1]
                ).result
                tv_b = pto.MakeTensorViewOp(
                    tv2_f32, arg_b, [c32, c32], [c32, c1]
                ).result
                tv_c = pto.MakeTensorViewOp(
                    tv2_f32, arg_c, [c32, c32], [c32, c1]
                ).result
                tv_out = pto.MakeTensorViewOp(
                    tv2_f32, arg_out, [c32, c32], [c32, c1]
                ).result

                sv_a = pto.PartitionViewOp(
                    tile_view_32, tv_a, offsets=[c0, c0], sizes=[c32, c32]
                ).result
                sv_b = pto.PartitionViewOp(
                    tile_view_32, tv_b, offsets=[c0, c0], sizes=[c32, c32]
                ).result
                sv_c = pto.PartitionViewOp(
                    tile_view_32, tv_c, offsets=[c0, c0], sizes=[c32, c32]
                ).result
                tb_a = pto.AllocTileOp(tile_buf_32).result
                tb_b = pto.AllocTileOp(tile_buf_32).result
                tb_c = pto.AllocTileOp(tile_buf_32).result

                pto.TLoadOp(None, sv_a, tb_a)
                pto.TLoadOp(None, sv_b, tb_b)
                pto.TLoadOp(None, sv_c, tb_c)

                def store_result(idx, tile):
                    offset = arith.ConstantOp(index, idx * 32 * 32).result
                    out_ptr = pto.AddPtrOp(arg_out, offset).result
                    out_tv = pto.MakeTensorViewOp(
                        tv2_f32, out_ptr, [c32, c32], [c32, c1]
                    ).result
                    out_sv = pto.PartitionViewOp(
                        tile_view_32, out_tv, offsets=[c0, c0], sizes=[c32, c32]
                    ).result
                    pto.TStoreOp(None, tile, out_sv)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TAddOp(tb_a, tb_b, tb_out)
                store_result(0, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TSubOp(tb_a, tb_b, tb_out)
                store_result(1, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TMulOp(tb_a, tb_b, tb_out)
                store_result(2, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TDivOp(tb_a, tb_b, tb_out)
                store_result(3, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TMaxOp(tb_a, tb_b, tb_out)
                store_result(4, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TMinOp(tb_a, tb_b, tb_out)
                store_result(5, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TRemOp(tb_a, tb_b, tb_out)
                store_result(6, tb_out)

                tb_tmp = pto.AllocTileOp(tile_buf_u8).result
                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TPReluOp(tb_a, tb_b, tb_tmp, tb_out)
                store_result(7, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TAddSOp(tb_a, scale, tb_out)
                store_result(8, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TSubSOp(tb_a, scale, tb_out)
                store_result(9, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TMulSOp(tb_a, scale, tb_out)
                store_result(10, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TDivSOp(tb_a, scale, tb_out)
                store_result(11, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TDivSOp(scale, tb_a, tb_out)
                store_result(12, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TMaxSOp(tb_a, scale, tb_out)
                store_result(13, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TMinSOp(tb_a, scale, tb_out)
                store_result(14, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TRemSOp(tb_a, scale, tb_out)
                store_result(15, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TAddCOp(tb_a, tb_b, tb_c, tb_out)
                store_result(16, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TSubCOp(tb_a, tb_b, tb_c, tb_out)
                store_result(17, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TAddSCOp(tb_a, bias, tb_b, tb_out)
                store_result(18, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TSubSCOp(tb_a, bias, tb_b, tb_out)
                store_result(19, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TAbsOp(tb_a, tb_out)
                store_result(20, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TNegOp(tb_a, tb_out)
                store_result(21, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TExpOp(tb_a, tb_out)
                store_result(22, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TLogOp(tb_a, tb_out)
                store_result(23, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TSqrtOp(tb_a, tb_out)
                store_result(24, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TRsqrtOp(tb_a, tb_out)
                store_result(25, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TRecipOp(tb_a, tb_out)
                store_result(26, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TReluOp(tb_a, tb_out)
                store_result(27, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TLReluOp(tb_a, slope, tb_out)
                store_result(28, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TPartAddOp(tb_a, tb_b, tb_out)
                store_result(29, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TPartMaxOp(tb_a, tb_b, tb_out)
                store_result(30, tb_out)

                tb_out = pto.AllocTileOp(tile_buf_32).result
                pto.TPartMinOp(tb_a, tb_b, tb_out)
                store_result(31, tb_out)

                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
