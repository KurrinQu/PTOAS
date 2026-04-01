// RUN: ptoas %s --enable-op-fusion --pto-backend=vpto --pto-arch=a5 --post-fusion-loop-unroll-factor=4 --print-ir-after-all --print-ir-after-all-func-filter=post_fusion_unroll_pipeline -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After PTOFusionLoadStoreElision/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s --check-prefix=LS
// RUN: awk '/IR Dump After PTOPostFusionLoopUnroll/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s --check-prefix=UNROLL
// RUN: awk '/IR Dump After PTOFlattenFusionRegion/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s --check-prefix=FLAT

// LS-LABEL: func.func @post_fusion_unroll_pipeline(
// LS: pto.fusion_region {
// LS: scf.for %{{[^ ]+}} = %c0 to %c32 step %c1 {
// LS: pto.vlds %0[%{{[^]]+}}]
// LS: pto.vmul
// LS: pto.yield(%14)

// UNROLL-LABEL: func.func @post_fusion_unroll_pipeline(
// UNROLL: pto.fusion_region {
// UNROLL: %c4 = arith.constant 4 : index
// UNROLL: scf.for %{{[^ ]+}} = %c0 to %c32 step %c4 {
// UNROLL: pto.vlds %0[%{{[^]]+}}]
// UNROLL: pto.vsts %{{[^,]+}}, %14[%{{[^]]+}}], %{{[^ ]+}}
// UNROLL: %c3 = arith.constant 3 : index
// UNROLL: pto.vlds %0[%{{[^]]+}}]
// UNROLL: pto.vsts %{{[^,]+}}, %14[%{{[^]]+}}], %{{[^ ]+}}

// FLAT-LABEL: func.func @post_fusion_unroll_pipeline(
// FLAT-NOT: pto.fusion_region
// FLAT-NOT: pto.yield
// FLAT: scf.for %{{[^ ]+}} = %c0 to %c32 step %c4 {
// FLAT: pto.vlds %0[%{{[^]]+}}]
// FLAT: pto.vsts %{{[^,]+}}, %7[%{{[^]]+}}], %mask

module {
  func.func @post_fusion_unroll_pipeline(%in0: !pto.ptr<f32>, %out0: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index

    %in_tv = pto.make_tensor_view %in0, shape = [%c32, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %out_tv = pto.make_tensor_view %out0, shape = [%c32, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %in_pt = pto.partition_view %in_tv, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %out_pt = pto.partition_view %out_tv, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

    %a = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %b = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %c = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tload ins(%in_pt : !pto.partition_tensor_view<32x32xf32>) outs(%a : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tadd ins(%a, %a : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tmul ins(%b, %a : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%out_pt : !pto.partition_tensor_view<32x32xf32>)
    return
  }
}
