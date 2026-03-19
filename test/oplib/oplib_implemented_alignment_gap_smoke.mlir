// RUN: ptoas %s --pto-arch=a5 --pto-level=level3 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=implemented_alignment_gap_smoke -o /dev/null > %t.out 2>&1 || true
// RUN: FileCheck %s < %t.out
// XFAIL: *

module {
  // CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
  // CHECK-LABEL: func.func @implemented_alignment_gap_smoke()
  // CHECK-DAG: call @__pto_oplib_inst_l3_float_unary_template_tabs{{.*}}(
  // CHECK-DAG: call @__pto_oplib_inst_l3_float_tile_scalar_template_tsubs{{.*}}(
  // CHECK-DAG: call @__pto_oplib_inst_l3_float_partial_binary_template_tpartmin{{.*}}(
  func.func @implemented_alignment_gap_smoke() {
    %f32_scale = arith.constant 1.500000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64

    %src0 = pto.alloc_tile addr = %c0_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %src1 = pto.alloc_tile addr = %c4096_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %dst = pto.alloc_tile addr = %c8192_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tabs ins(%src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tsubs ins(%dst, %f32_scale : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f32) outs(%src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tpartmin ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
