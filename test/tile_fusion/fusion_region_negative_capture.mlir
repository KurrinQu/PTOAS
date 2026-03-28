// RUN: { ptoas %s -o /dev/null 2>&1 || true; } | FileCheck %s

// Negative 5.5 minimal-container regression:
// a fusion_region body is allowed to capture external SSA values implicitly,
// but it must not declare explicit body block arguments.

module {
  func.func @fusion_region_negative_capture(
      %arg0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {
    "pto.fusion_region"() ({
    ^bb0(%captured: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>):
      %0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      pto.tmov ins(%captured : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      "pto.yield"() : () -> ()
    }) : () -> ()
    return
  }
}

// CHECK: error: 'pto.fusion_region' op expects body block to have no arguments, got 1
// CHECK: Error: Failed to parse MLIR.
