module attributes {pto.target_arch = "a5"} {
  func.func @matmul_tpush_tpop_print(
      %gm_a: !pto.ptr<f32>,
      %gm_b: !pto.ptr<f32>,
      %c2v_consumer_buf: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    // TileBuf-only tprint path (currently may be folded out by some lowering passes).
    %vec_print = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %mat_a_tile = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %mat_b_tile = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %left_tile = pto.alloc_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %right_tile = pto.alloc_tile : !pto.tile_buf<loc=right, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>
    %acc_tile = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>

    %gm_a_view = pto.make_tensor_view %gm_a, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<?x?xf32>
    %gm_b_view = pto.make_tensor_view %gm_b, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<?x?xf32>
    %gm_a_tile_view = pto.partition_view %gm_a_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<16x16xf32>
    %gm_b_tile_view = pto.partition_view %gm_b_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<16x16xf32>

    %pipe = pto.initialize_l2l_pipe {dir_mask = 1}
      (%c2v_consumer_buf : i32)
      -> !pto.pipe<
           !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>,
           !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>>

    pto.section.cube {
      pto.tload ins(%gm_a_tile_view : !pto.partition_tensor_view<16x16xf32>) outs(%mat_a_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
      pto.tload ins(%gm_b_tile_view : !pto.partition_tensor_view<16x16xf32>) outs(%mat_b_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>)

      pto.tmov ins(%mat_a_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>) outs(%left_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
      pto.tmov ins(%mat_b_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>) outs(%right_tile : !pto.tile_buf<loc=right, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>)

      pto.tmatmul ins(%left_tile, %right_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>, !pto.tile_buf<loc=right, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>) outs(%acc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)

      pto.tpush(%acc_tile, %pipe : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>>)
    }

    pto.section.vector {
      %slot_id = pto.tpop(%pipe : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>>) -> index
      %fifo_tile = pto.get_fifo_tile(%pipe, %slot_id : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>>, index)
        -> !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      pto.tmov ins(%fifo_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%vec_print : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tprint ins(%vec_print : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tfree(%pipe, %slot_id : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>>, index)
    }

    return
  }
}
