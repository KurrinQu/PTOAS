// RUN: not ptoas --pto-arch=a5 %s 2>&1 | FileCheck %s
//
// One pipe may not have two outstanding borrowed tiles at the same time, even
// when both slots eventually have explicit tfree.

module {
  func.func @multiple_outstanding_same_pipe(
      %c2v_consumer_buf: i32) {
    %vec_tile0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %vec_tile1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    %pipe = pto.initialize_l2l_pipe {dir_mask = 1}
      (%c2v_consumer_buf : i32)
      -> !pto.pipe<
           !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>,
           !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>

    pto.section.vector {
      %slot_id0 = pto.tpop(%pipe : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>) -> index
      %fifo_tile0 = pto.get_fifo_tile(%pipe, %slot_id0 : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>, index)
        -> !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>

      %slot_id1 = pto.tpop(%pipe : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>) -> index
      %fifo_tile1 = pto.get_fifo_tile(%pipe, %slot_id1 : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>, index)
        -> !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>

      // The first borrowed tile stays live past the second pop.
      pto.tabs ins(%fifo_tile0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%vec_tile0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tfree(%pipe, %slot_id0 : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>, index)
      pto.tmov ins(%fifo_tile1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%vec_tile1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tfree(%pipe, %slot_id1 : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>, index)
    }

    return
  }
}

// CHECK: error:
// CHECK: multiple outstanding pops on the same pipe are not supported
