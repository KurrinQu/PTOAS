// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir %s -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @tbinary_tail_mask_select
// CHECK: %[[MASK:.*]], %[[SCALAR_OUT:.*]] = a5vm.plt_b32
// CHECK: %[[LHS:.*]] = a5vm.vlds
// CHECK: %[[RHS:.*]] = a5vm.vlds
// CHECK: %[[BIN:.*]] = a5vm.vmax %[[LHS]], %[[RHS]], %[[MASK]]
// CHECK: a5vm.vsts %[[BIN]], %{{.+}}, %[[MASK]]

module {
  func.func @tbinary_tail_mask_select() {
    %src0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1, v_row=16, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %src1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1, v_row=16, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %dst = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1, v_row=16, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tmax ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1, v_row=16, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1, v_row=16, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1, v_row=16, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
