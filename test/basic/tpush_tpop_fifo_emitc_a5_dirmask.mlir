// RUN: ptoas --pto-arch=a5 %s | FileCheck %s
// RUN: ptoas --pto-arch=a5 --enable-insert-sync %s 2>&1 | FileCheck %s --check-prefix=SYNC

module {
  func.func @pipe_emitc_a5_dirmask(
      %c2v_consumer_buf: i32,
      %v2c_consumer_buf: i32) {
    %acc_tile = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    %cube_tile = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %vec_tile = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    %pipe_c2v = pto.initialize_l2l_pipe {dir_mask = 1}
      (%c2v_consumer_buf : i32)
      -> !pto.pipe<
           !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>,
           !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>

    %pipe_v2c = pto.initialize_l2l_pipe {dir_mask = 2}
      (%v2c_consumer_buf : i32)
      -> !pto.pipe<
           !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
           !pto.tile_buf<loc=mat, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>>

    pto.section.cube {
      pto.tpush(%acc_tile, %pipe_c2v : !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>)
      %slot_id_v2c = pto.tpop(%pipe_v2c : !pto.pipe<!pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=mat, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>>) -> index
      %cube_fifo_tile = pto.get_fifo_tile(%pipe_v2c, %slot_id_v2c : !pto.pipe<!pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=mat, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>>, index)
        -> !pto.tile_buf<loc=mat, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>
      pto.tmov ins(%cube_fifo_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>) outs(%cube_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
      pto.tfree(%pipe_v2c, %slot_id_v2c : !pto.pipe<!pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=mat, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>>, index)
    }

    pto.section.vector {
      %slot_id_c2v = pto.tpop(%pipe_c2v : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>) -> index
      %vec_fifo_tile = pto.get_fifo_tile(%pipe_c2v, %slot_id_c2v : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>, index)
        -> !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      pto.tmov ins(%vec_fifo_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%vec_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tfree(%pipe_c2v, %slot_id_c2v : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>, index)
      pto.tpush(%vec_tile, %pipe_v2c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.pipe<!pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=mat, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>>)
    }
    return
  }
}

// CHECK: __global__ AICORE void pipe_emitc_a5_dirmask
// CHECK: auto {{.*}} = TPipe<0, FIFOType::VEC_FIFO
// CHECK-SAME: Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null>
// CHECK: auto {{.*}} = TPipe<2, FIFOType::MAT_FIFO
// CHECK-SAME: Tile<TileType::Vec
// CHECK-SAME: Tile<TileType::Mat, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null>
// CHECK-DAG: TPUSH(
// CHECK-DAG: TPOP(
// CHECK-DAG: TFREE(

// SYNC-NOT: assigned_pipe
// SYNC: pto.tpop_internal(
// SYNC: pto.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
// SYNC: pto.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
