from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            idx = IndexType.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_16 = pto.PartitionTensorViewType.get([16, 16], f32, ctx)
            tile_view_32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)

            src_ty = pto.TileBufType.get([16, 16], f32, vec, [16, 16], cfg, ctx)
            dst_ty = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("assemble_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c8 = arith.ConstantOp(idx, 8).result
                c16 = arith.ConstantOp(idx, 16).result
                c32 = arith.ConstantOp(idx, 32).result

                arg_src, arg_dst, arg_out = entry.arguments

                tv_src = pto.MakeTensorViewOp(tv2_f32, arg_src, [c16, c16], [c16, c1]).result
                tv_dst = pto.MakeTensorViewOp(tv2_f32, arg_dst, [c32, c32], [c32, c1]).result
                tv_out = pto.MakeTensorViewOp(tv2_f32, arg_out, [c32, c32], [c32, c1]).result

                sv_src = pto.PartitionViewOp(tile_view_16, tv_src, offsets=[c0, c0], sizes=[c16, c16]).result
                sv_dst = pto.PartitionViewOp(tile_view_32, tv_dst, offsets=[c0, c0], sizes=[c32, c32]).result
                sv_out = pto.PartitionViewOp(tile_view_32, tv_out, offsets=[c0, c0], sizes=[c32, c32]).result

                src = pto.AllocTileOp(src_ty).result
                dst = pto.AllocTileOp(dst_ty).result

                pto.TLoadOp(None, sv_src, src)
                pto.TLoadOp(None, sv_dst, dst)
                pto.TAssembleOp(src, c8, c8, dst)
                pto.TStoreOp(None, dst, sv_out)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
