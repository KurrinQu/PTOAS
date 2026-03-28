// RUN: { ptoas %s --test-only-fusion-region-gen -o /dev/null 2>&1 || true; } | FileCheck %s

// Negative 5.5 split-span regression:
// when the same pto.fusion.group_id appears in two discontiguous spans within
// one basic block, PTOFusionRegionGen must fail explicitly.

module {
  func.func @fusion_region_negative_split() {
    %0 = pto.alloc_tile {pto.fusion.group_id = 7 : i64, pto.fusion.order = 0 : i64} : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %2 = pto.alloc_tile {pto.fusion.group_id = 7 : i64, pto.fusion.order = 1 : i64} : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    return
  }
}

// CHECK: error: expected one contiguous span per pto.fusion.group_id within a basic block
// CHECK: Error: Pass execution failed.
