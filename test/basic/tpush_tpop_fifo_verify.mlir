// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @verify_bidirectional_dir_mask_not_supported(
      %gm_slot_buffer: !pto.ptr<f32>) {
    %acc_tile = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    %pipe = pto.initialize_l2g2l_pipe {dir_mask = 3}
      (%gm_slot_buffer : !pto.ptr<f32>)
      -> !pto.pipe<
           !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>,
           !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>

    pto.section.cube {
      pto.tpush(%acc_tile, %pipe : !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>)
    }
    return
  }
}

// CHECK: error: 'pto.initialize_l2g2l_pipe' op dir_mask must be 1 (C2V) or 2 (V2C)
