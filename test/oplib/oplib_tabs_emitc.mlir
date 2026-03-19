// RUN: ptoas %s --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --pto-level=level3 -o %t.cpp
// RUN: FileCheck %s --check-prefix=EMITC < %t.cpp
// XFAIL: *

// EMITC-LABEL: __global__ AICORE void tabs_emitc_smoke(
// EMITC-DAG: __VEC_SCOPE__ {
// EMITC-DAG: CreatePredicate<float>(
// EMITC-DAG: vneg(
// EMITC-DAG: vmax(
// EMITC-NOT: failed to legalize operation 'arith.maximumf'

module {
  func.func @tabs_emitc_smoke() {
    %c0_i64 = arith.constant 0 : i64
    %c4096_i64 = arith.constant 4096 : i64

    %src = pto.alloc_tile addr = %c0_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %dst = pto.alloc_tile addr = %c4096_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tabs ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
