// RUN: ./build/tools/ptoas/ptoas --pto-backend=vpto --emit-vpto %s -o - 2>/dev/null | FileCheck %s

// CHECK-LABEL: func.func @tabs_abs_loop_shape
// CHECK: %[[BASE:[^ ]+]] = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>
// CHECK: pto.vecscope {
// CHECK: scf.for %[[CHUNK:[^ ]+]] = %{{[^ ]+}} to %{{[^ ]+}} step %{{[^ ]+}}
// CHECK: pto.vlds
// CHECK: pto.vabs
// CHECK: pto.vsts
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: pto.scope = "__VEC_SCOPE__"
// CHECK-NOT: dist = "__VEC_SCOPE__"
// CHECK-NOT: emitc.call_opaque "TABS"

module {
  func.func @tabs_abs_loop_shape() {
    %src = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %dst = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tabs ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}

// The chosen lowered vector interval is represented as an explicit
// pto.vecscope region instead of a dummy carrier loop.
// This contract therefore locks both scope ownership and the ordered
// pto.vlds -> pto.vabs -> pto.vsts vector primitive sequence.
