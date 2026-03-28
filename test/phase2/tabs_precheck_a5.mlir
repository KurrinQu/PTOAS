// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: abs lowering requires tile domain vec
// CHECK: abs lowering requires row-major tile layout
// CHECK: abs lowering requires destination valid rows not to exceed source
// CHECK: abs lowering supports only f16 and f32 element types
// CHECK-NOT: failed to legalize operation
// CHECK-NOT: a5vm.copy_gm_to_ubuf
// CHECK-NOT: a5vm.vlds
// CHECK-NOT: a5vm.vabs
// CHECK-NOT: a5vm.vsts
// CHECK-NOT: a5vm.copy_ubuf_to_gm

module {
  func.func @tabs_bad_domain() {
    %src = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=row_major, fractal=512, pad=0>
    %dst = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tabs ins(%src : !pto.tile_buf<loc=mat, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=row_major, fractal=512, pad=0>)
      outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }

  func.func @tabs_bad_layout() {
    %src = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=col_major, slayout=none_box, fractal=512, pad=0>
    %dst = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tabs ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=col_major, slayout=none_box, fractal=512, pad=0>)
      outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }

  func.func @tabs_bad_valid_region() {
    %src = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=16, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %dst = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tabs ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=16, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }

  func.func @tabs_bad_dtype() {
    %src = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %dst = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tabs ins(%src : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
