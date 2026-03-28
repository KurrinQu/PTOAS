// RUN: ptoas %s --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1 | FileCheck %s
// XFAIL: *

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-LABEL: func.func @compare_mask_ui8_compat(
// CHECK: call @__pto_oplib_inst_l3_cmp_tile_scalar_template_tcmps_gt(
// CHECK-NOT: E_OPLIB_SIMD_UNSUPPORTED_LAYOUT

module {
  func.func @compare_mask_ui8_compat(%in0: !pto.ptr<f32>, %out0: !pto.ptr<ui8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %threshold = arith.constant 1.000000e+00 : f32

    %in0_tv = pto.make_tensor_view %in0, shape = [%c32, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %out0_tv = pto.make_tensor_view %out0, shape = [%c32, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xui8>

    %in0_pt = pto.partition_view %in0_tv, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %out0_pt = pto.partition_view %out0_tv, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<?x?xui8> -> !pto.partition_tensor_view<32x32xui8>

    %src = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %mask = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=ui8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tload ins(%in0_pt : !pto.partition_tensor_view<32x32xf32>) outs(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tcmps ins(%src, %threshold {cmpMode = #pto<cmp gt>} : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f32) outs(%mask : !pto.tile_buf<loc=vec, dtype=ui8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%mask : !pto.tile_buf<loc=vec, dtype=ui8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%out0_pt : !pto.partition_tensor_view<32x32xui8>)
    return
  }
}
