// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @verify_get_fifo_tile_pipe_mismatch(
      %gm_slot_buffer: !pto.ptr<f32>,
      %local_addr0: i32,
      %local_addr1: i32) {
    %pipe0 = pto.initialize_l2g2l_pipe {dir_mask = 1}
      (%gm_slot_buffer : !pto.ptr<f32>, %local_addr0 : i32)
      -> !pto.pipe<
           !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>,
           !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>
    %pipe1 = pto.initialize_l2l_pipe {dir_mask = 1}
      (%local_addr1 : i32)
      -> !pto.pipe<
           !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>,
           !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>

    pto.section.vector {
      %slot_id = pto.tpop(%pipe0 : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>) -> index
      %tile = pto.get_fifo_tile(%pipe1, %slot_id : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>, index)
        -> !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      pto.tfree(%pipe0, %slot_id : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>, index)
    }
    return
  }
}

// CHECK: error: 'pto.get_fifo_tile' op pipe_handle must match the pto.tpop that produced slot_id
