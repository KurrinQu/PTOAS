// RUN: { ptoas %s --test-only-fusion-region-gen --print-ir-after-all --print-ir-after-all-func-filter=fusion_region_internal_dead_omission -o /dev/null 2>&1 || true; } | awk '/IR Dump After PTOFusionRegionGen/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOFusionRegionGen/) exit; print}' | FileCheck %s
// RUN: { ptoas %s --enable-op-fusion --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=fusion_region_internal_dead_omission -o /dev/null 2>&1 || true; } | awk '/IR Dump After PTOFusionLoadStoreElision/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOFusionLoadStoreElision/) exit; print}' | FileCheck %s --check-prefix=ELIDE

// Region frontier regression:
// a DPS destination reused by a later overwrite must not force an internal-dead
// earlier write-instance into pto.yield / region results. The shared storage
// stays outside the region for the later overwrite, while only the truly
// escaping write is yielded.
// Store-elision follow-up:
// once the region is lowered, the final maskedstore for the later non-yielded
// overwrite of that reused storage must disappear, while the yielded frontier
// store for `%sum` remains materialized.

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

// ELIDE-LABEL: IR Dump After PTOFusionLoadStoreElision
// ELIDE-LABEL: func.func @fusion_region_internal_dead_omission(
// ELIDE: %[[REGION:[0-9]+]] = pto.fusion_region {
// ELIDE: %[[TMP:[0-9]+]] = pto.bind_tile
// ELIDE: %[[SUM:[0-9]+]] = pto.bind_tile
// ELIDE: %[[TMP_EARLY_MEM:[0-9]+]] = pto.simd.tile_to_memref %[[TMP]]
// ELIDE: vector.maskedstore %[[TMP_EARLY_MEM]]
// ELIDE: %[[SUM_MEM:[0-9]+]] = pto.simd.tile_to_memref %[[SUM]]
// ELIDE: vector.maskedstore %[[SUM_MEM]]
// ELIDE: %[[FINAL_MUL:[0-9]+]] = arith.mulf %{{[0-9]+}}, %{{[0-9]+}}
// ELIDE-NOT: vector.maskedstore
// ELIDE: pto.yield(%[[SUM]]) :
// ELIDE: } {pto.fusion.group_id = 0 : i64} :
