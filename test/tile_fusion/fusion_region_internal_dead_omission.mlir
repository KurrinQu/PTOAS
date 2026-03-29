// RUN: { ptoas %s --test-only-fusion-region-gen --print-ir-after-all --print-ir-after-all-func-filter=fusion_region_internal_dead_omission -o /dev/null 2>&1 || true; } | awk '/IR Dump After PTOFusionRegionGen/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOFusionRegionGen/) exit; print}' | FileCheck %s

// Region frontier regression:
// a DPS destination reused by a later overwrite must not force an internal-dead
// earlier write-instance into pto.yield / region results. The shared storage
// stays outside the region for the later overwrite, while only the truly
// escaping write is yielded.
// This test intentionally stops at FusionRegionGen. The later VPTO lowering
// path still rejects the dynamic-valid trowexpandmul in this fixture, so
// post-lowering store-elision checks are obsolete here.

module {
  func.func @fusion_region_internal_dead_omission(
      %full0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %full1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %full2: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %full3: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %row0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scratch: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {
    %tmp = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %sum = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %sink = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    "pto.trowexpandmul"(%full0, %row0, %tmp) {pto.fusion.group_id = 0 : i64, pto.fusion.order = 0 : i64} : (!pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) -> ()
    "pto.tadd"(%tmp, %full1, %sum) {pto.fusion.group_id = 0 : i64, pto.fusion.order = 1 : i64} : (!pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) -> ()
    "pto.tmul"(%full2, %full3, %tmp) : (!pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) -> ()
    "pto.ttrans"(%sum, %scratch, %sink) : (!pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) -> ()
    return
  }
}

// CHECK-LABEL: IR Dump After PTOFusionRegionGen
// CHECK-LABEL: func.func @fusion_region_internal_dead_omission(
// CHECK: %[[TMP:[0-9]+]] = pto.alloc_tile
// CHECK: %[[SINK:[0-9]+]] = pto.alloc_tile
// CHECK: %[[REGION:[0-9]+]] = pto.fusion_region {
// CHECK: %[[SUM:[0-9]+]] = pto.alloc_tile
// CHECK: pto.trowexpandmul ins(%arg0, %arg4
// CHECK-SAME: outs(%[[TMP]]
// CHECK: pto.tadd ins(%[[TMP]], %arg1
// CHECK-SAME: outs(%[[SUM]]
// CHECK-NOT: pto.yield(%[[TMP]]
// CHECK: pto.yield(%[[SUM]]) :
// CHECK: } {pto.fusion.group_id = 0 : i64} :
// CHECK: pto.tmul ins(%arg2, %arg3
// CHECK-SAME: outs(%[[TMP]]
// CHECK: pto.ttrans ins(%[[REGION]], %arg5
// CHECK-SAME: outs(%[[SINK]]
// CHECK: return
