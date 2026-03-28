// RUN: { ptoas %s --test-only-fusion-region-gen -o /dev/null 2>&1 || true; } | FileCheck %s

// Negative 5.5 metadata regression:
// incomplete pto.fusion.* metadata must be rejected explicitly.

module {
  func.func @fusion_region_negative_metadata() {
    %0 = pto.alloc_tile {pto.fusion.group_id = 3 : i64} : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    return
  }
}

// CHECK: error: expected pto.fusion.group_id and pto.fusion.order to either both exist or both be absent
// CHECK: Error: Pass execution failed.
