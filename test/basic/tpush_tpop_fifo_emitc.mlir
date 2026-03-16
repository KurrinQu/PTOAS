// RUN: ptoas %s | FileCheck %s

module {
  func.func @pipe_emitc(
      %gm_slot_buffer: !pto.ptr<f32>,
      %local_fifo_addr: i32) {
    %acc_tile = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    %vec_tile = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    %pipe = pto.initialize_l2g2l_pipe {dir_mask = 1, local_fifo_depth = 4}
      (%gm_slot_buffer : !pto.ptr<f32>, %local_fifo_addr : i32)
      -> !pto.pipe<
           !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>,
           !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>

    pto.section.cube {
      pto.tpush(%acc_tile, %pipe : !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>)
    }

    pto.section.vector {
      %slot_id = pto.tpop(%pipe : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>) -> index
      %fifo_tile = pto.get_fifo_tile(%pipe, %slot_id : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>, index)
        -> !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      pto.tmov ins(%fifo_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%vec_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tfree(%pipe, %slot_id : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>, index)
    }
    return
  }
}

// CHECK: #include "pto/pto-inst.hpp"
// CHECK-NOT: memref<
// CHECK: __global__ AICORE void pipe_emitc
// CHECK: auto {{.*}} = TPipe<0, FIFOType::GM_FIFO
// CHECK-SAME: Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null>
// CHECK-SAME: Tile<TileType::Vec
// CHECK-SAME: false, 4
// CHECK: TPUSH(
// CHECK: TPOP(
// CHECK: TFREE(
