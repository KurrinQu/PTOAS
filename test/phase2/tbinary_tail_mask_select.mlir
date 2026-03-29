// RUN: ./build/tools/ptoas/ptoas --pto-backend=vpto --vpto-print-ir %s -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @tbinary_tail_mask_select
// CHECK: %[[MASK:.*]], %[[SCALAR_OUT:.*]] = pto.plt_b32
// CHECK: %[[LHS:.*]] = pto.vlds
// CHECK: %[[RHS:.*]] = pto.vlds
// CHECK: %[[BIN:.*]] = pto.vmax %[[LHS]], %[[RHS]], %[[MASK]]
// CHECK: pto.vsts %[[BIN]], %{{.+}}, %[[MASK]]

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
