// RUN: ptoas %s -o /dev/null

module {
  func.func @tmuls_f16(
      %src: !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: f16,
      %dst: !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {
    pto.tmuls ins(%src, %scalar : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f16)
      outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }

  func.func @tmuls_i16(
      %src: !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {
    pto.tmuls ins(%src, %scalar : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, i16)
      outs(%dst : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }

  func.func @tmuls_i32(
      %src: !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i32,
      %dst: !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {
    pto.tmuls ins(%src, %scalar : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, i32)
      outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
