// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir %s -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @tabs_abs_loop_shape
// CHECK: scf.for
// CHECK: scf.for
// CHECK: a5vm.vlds
// CHECK: a5vm.vabs
// CHECK: a5vm.vsts
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

// The vec-scope is represented structurally instead of as a dedicated A5VM op.
// This contract therefore locks the surrounding loop nesting and ordered vector
// primitive sequence rather than a standalone vec-scope mnemonic.
