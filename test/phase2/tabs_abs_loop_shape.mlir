// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir %s -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @tabs_abs_loop_shape
// CHECK: scf.for %[[DUMMY:[^ ]+]] = %{{[^ ]+}} to %{{[^ ]+}} step %{{[^ ]+}} {
// CHECK: scf.for %[[CHUNK:[^ ]+]] = %{{[^ ]+}} to %{{[^ ]+}} step %{{[^ ]+}}
// CHECK: a5vm.vlds
// CHECK: a5vm.vabs
// CHECK: a5vm.vsts
// CHECK: } {a5vm.scope = "__VEC_SCOPE__", cce_aiv_loop_hint, llvm.loop.aivector_scope
// CHECK: } {a5vm.loop_scope_depth = 0 : i64, a5vm.scope = "__VEC_SCOPE__", cce_aiv_loop_hint, llvm.loop.aivector_scope
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

// The chosen lowered loop carries explicit AIV vec-scope semantics through
// cce_aiv_loop_hint before lowering and llvm.loop.aivector_scope after lowering.
// This contract therefore locks both loop ownership and the ordered
// a5vm.vlds -> a5vm.vabs -> a5vm.vsts vector primitive sequence.
