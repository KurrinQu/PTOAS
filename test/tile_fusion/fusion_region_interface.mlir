// RUN: { ptoas %s --enable-op-fusion --pto-backend=a5vm --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=fusion_region_interface -o /dev/null 2>&1 || true; } | awk '/IR Dump After PTOFusionRegionGen/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOFusionRegionGen/) exit; print}' | FileCheck %s --check-prefix=GEN
// RUN: { ptoas %s --enable-op-fusion --pto-backend=a5vm --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=fusion_region_interface -o /dev/null 2>&1 || true; } | awk '/IR Dump After .*PTOViewToMemrefPass/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOViewToMemrefPass/) exit; print}' | FileCheck %s --check-prefix=VIEW
// RUN: { ptoas %s --enable-op-fusion --pto-backend=a5vm --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=fusion_region_interface -o /dev/null 2>&1 || true; } | awk '/IR Dump After PlanMemory/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PlanMemory/) exit; print}' | FileCheck %s --check-prefix=PLAN
// RUN: { ptoas %s --enable-op-fusion --pto-backend=a5vm --enable-insert-sync --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=fusion_region_interface -o /dev/null 2>&1 || true; } | awk '/IR Dump After PTOInsertSync/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOInsertSync/) exit; print}' | FileCheck %s --check-prefix=SYNC

// Region interface regression:
// DPS destination tiles that remain externally visible after the fused span
// must become explicit pto.fusion_region results / pto.yield operands, while
// external inputs stay implicitly captured and scratch alloc_tile temporaries
// can be sunk into the region body. This also makes pto.yield the explicit
// summary of which internal tiles still escape the region boundary.
// This regression is intentionally scoped to the pre-lowering seam. The A5VM
// lowering path still rejects the dynamic-valid trowexpandmul instances in this
// fixture, and the old LIB/INLINE/LOW/ELIDE stages were checking a retired
// pipeline branch.

module {
  func.func @fusion_region_interface(
      %full0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %full1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %row0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %row1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {
    %tmp0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %tmp1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %sum = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %transTmp0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %transTmp1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %sink0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %sink1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.trowexpandmul ins(%full0, %row0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.trowexpandmul ins(%full1, %row1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tadd ins(%tmp0, %tmp1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%sum : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    pto.ttrans ins(%tmp0, %transTmp0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%sink0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.ttrans ins(%sum, %transTmp1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%sink1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}

// GEN-LABEL: IR Dump After PTOFusionRegionGen
// GEN-LABEL: func.func @fusion_region_interface(
// GEN: %[[REGION:.*]]:2 = pto.fusion_region {
// GEN: %[[TMP0:[0-9]+]] = pto.alloc_tile
// GEN: %[[TMP1:[0-9]+]] = pto.alloc_tile
// GEN: %[[SUM:[0-9]+]] = pto.alloc_tile
// GEN: pto.trowexpandmul ins(%arg0, %arg2
// GEN-SAME: outs(%[[TMP0]]
// GEN: pto.trowexpandmul ins(%arg1, %arg3
// GEN-SAME: outs(%[[TMP1]]
// GEN: pto.tadd ins(%[[TMP0]], %[[TMP1]] :
// GEN-SAME: outs(%[[SUM]]
// GEN: pto.yield(%[[TMP0]], %[[SUM]]) : (!pto.tile_buf
// GEN: } {pto.fusion.group_id = 0 : i64} : !pto.tile_buf
// GEN: pto.ttrans ins(%[[REGION]]#0, %{{.*}})
// GEN: pto.ttrans ins(%[[REGION]]#1, %{{.*}})
// GEN: return

// VIEW-LABEL: IR Dump After mlir::pto::{anonymous}::PTOViewToMemrefPass
// VIEW-LABEL: func.func @fusion_region_interface(
// VIEW: %[[REGION:.*]]:2 = pto.fusion_region {
// VIEW: %[[TMP0:[0-9]+]] = pto.bind_tile
// VIEW: %[[TMP1:[0-9]+]] = pto.bind_tile
// VIEW: %[[SUM:[0-9]+]] = pto.bind_tile
// VIEW: pto.trowexpandmul ins(%arg0, %arg2
// VIEW-SAME: outs(%[[TMP0]]
// VIEW: pto.trowexpandmul ins(%arg1, %arg3
// VIEW-SAME: outs(%[[TMP1]]
// VIEW: pto.tadd ins(%[[TMP0]], %[[TMP1]] :
// VIEW-SAME: outs(%[[SUM]]
// VIEW: pto.yield(%[[TMP0]], %[[SUM]]) : (memref
// VIEW: } {pto.fusion.group_id = 0 : i64} : memref
// VIEW: pto.ttrans ins(%[[REGION]]#0, %{{.*}})
// VIEW: pto.ttrans ins(%[[REGION]]#1, %{{.*}})
// VIEW: return

// PLAN-LABEL: IR Dump After PlanMemory
// PLAN-LABEL: func.func @fusion_region_interface(
// PLAN: %[[REGION:.*]]:2 = pto.fusion_region {
// PLAN: %[[TMP0:[0-9]+]] = pto.bind_tile
// PLAN: %[[TMP1:[0-9]+]] = pto.bind_tile
// PLAN: %[[SUM:[0-9]+]] = pto.bind_tile
// PLAN: pto.trowexpandmul ins(%arg0, %arg2
// PLAN-SAME: outs(%[[TMP0]]
// PLAN: pto.trowexpandmul ins(%arg1, %arg3
// PLAN-SAME: outs(%[[TMP1]]
// PLAN: pto.tadd ins(%[[TMP0]], %[[TMP1]] :
// PLAN-SAME: outs(%[[SUM]]
// PLAN: pto.yield(%[[TMP0]], %[[SUM]]) : (memref
// PLAN: } {pto.fusion.group_id = 0 : i64} : memref
// PLAN: pto.ttrans ins(%[[REGION]]#0, %{{.*}})
// PLAN: pto.ttrans ins(%[[REGION]]#1, %{{.*}})
// PLAN: return

// SYNC-LABEL: IR Dump After PTOInsertSync
// SYNC-LABEL: func.func @fusion_region_interface(
// SYNC: %[[REGION:.*]]:2 = pto.fusion_region {
// SYNC: %[[TMP0:[0-9]+]] = pto.bind_tile
// SYNC: %[[TMP1:[0-9]+]] = pto.bind_tile
// SYNC: %[[SUM:[0-9]+]] = pto.bind_tile
// SYNC: pto.trowexpandmul ins(%arg0, %arg2
// SYNC-SAME: outs(%[[TMP0]]
// SYNC: pto.trowexpandmul ins(%arg1, %arg3
// SYNC-SAME: outs(%[[TMP1]]
// SYNC: pto.tadd ins(%[[TMP0]], %[[TMP1]] :
// SYNC-SAME: outs(%[[SUM]]
// SYNC-NOT: pto.barrier
// SYNC-NOT: pto.record_event
// SYNC-NOT: pto.wait_event
// SYNC: pto.yield(%[[TMP0]], %[[SUM]]) : (memref
// SYNC: } {pto.fusion.group_id = 0 : i64} : memref
// SYNC: pto.ttrans ins(%[[REGION]]#0, %{{.*}})
// SYNC: pto.ttrans ins(%[[REGION]]#1, %{{.*}})
// SYNC: pto.barrier <PIPE_ALL> {pto.auto_sync_tail_barrier}
// SYNC: return
