// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @verify_gm_missing_fifo_addr() {
    %pipe = pto.initialize_l2g2l_pipe {dir_mask = 1}
      -> !pto.pipe<
           !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>,
           !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>
    return
  }
}

// CHECK: error: expected '('
