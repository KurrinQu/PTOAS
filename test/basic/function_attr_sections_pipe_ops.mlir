// RUN: ptoas --pto-arch=a5 %s 2>&1 | FileCheck %s --check-prefix=IR

module {
  func.func @cube_push(%c2v_consumer_buf: i32) attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %pipe = pto.initialize_l2l_pipe {
      dir_mask = 1,
      slot_size = 1024,
      slot_num = 8,
      flag_base = 0
    }(%c2v_consumer_buf : i32) -> !pto.pipe
    %acc_tile = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    pto.tpush_internal(%acc_tile, %pipe : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.pipe) {split = 0}
    return
  }

  func.func @vector_pop(%c2v_consumer_buf: i32) attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %pipe = pto.initialize_l2l_pipe {
      dir_mask = 1,
      slot_size = 1024,
      slot_num = 8,
      flag_base = 0
    }(%c2v_consumer_buf : i32) -> !pto.pipe
    %vec_tile = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %fifo_tile = pto.declare_tile -> !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tpop_internal(%fifo_tile, %pipe : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.pipe) {split = 1}
    pto.tmov ins(%fifo_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%vec_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tfree_internal(%pipe : !pto.pipe) {split = 2}
    return
  }
}

// IR-LABEL: func.func @cube_push(%arg0: i32) attributes {pto.kernel_kind = #pto.kernel_kind<cube>}
// IR: pto.section.cube {
// IR: %[[PIPE:.+]] = pto.initialize_l2l_pipe
// IR: pto.tpush_internal(

// IR-LABEL: func.func @vector_pop(%arg0: i32) attributes {pto.kernel_kind = #pto.kernel_kind<vector>}
// IR: pto.section.vector {
// IR: %[[PIPE:.+]] = pto.initialize_l2l_pipe
// IR: pto.tpop_internal(
// IR: pto.tfree_internal(
