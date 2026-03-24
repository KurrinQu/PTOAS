// RUN: { ptoas %s --enable-op-fusion --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=fusion_region_interface -o /dev/null 2>&1 || true; } | awk '/IR Dump After PTOFusionRegionGen/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOFusionRegionGen/) exit; print}' | FileCheck %s --check-prefix=GEN
// RUN: { ptoas %s --enable-op-fusion --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=fusion_region_interface -o /dev/null 2>&1 || true; } | awk '/IR Dump After .*PTOViewToMemrefPass/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOViewToMemrefPass/) exit; print}' | FileCheck %s --check-prefix=VIEW
// RUN: { ptoas %s --enable-op-fusion --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=fusion_region_interface -o /dev/null 2>&1 || true; } | awk '/IR Dump After PlanMemory/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PlanMemory/) exit; print}' | FileCheck %s --check-prefix=PLAN
// RUN: { ptoas %s --enable-op-fusion --enable-insert-sync --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=fusion_region_interface -o /dev/null 2>&1 || true; } | awk '/IR Dump After PTOInsertSync/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOInsertSync/) exit; print}' | FileCheck %s --check-prefix=SYNC
// RUN: { ptoas %s --enable-op-fusion --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=fusion_region_interface -o /dev/null 2>&1 || true; } | awk '/IR Dump After PTOInstantiateAndLowerToLibCall/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOInstantiateAndLowerToLibCall/) exit; print}' | FileCheck %s --check-prefix=LIB
// RUN: { ptoas %s --enable-op-fusion --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=fusion_region_interface -o /dev/null 2>&1 || true; } | awk '/IR Dump After PTOInlineLibCall/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOInlineLibCall/) exit; print}' | FileCheck %s --check-prefix=INLINE
// RUN: { ptoas %s --enable-op-fusion --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=fusion_region_interface -o /dev/null 2>&1 || true; } | awk '/IR Dump After PTOLowLevelLoopFusion/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOLowLevelLoopFusion/) exit; print}' | FileCheck %s --check-prefix=LOW
// RUN: { ptoas %s --enable-op-fusion --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=fusion_region_interface -o /dev/null 2>&1 || true; } | awk '/IR Dump After PTOFusionLoadStoreElision/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOFusionLoadStoreElision/) exit; print}' | FileCheck %s --check-prefix=ELIDE

// Region interface regression:
// DPS destination tiles that remain externally visible after the fused span
// must become explicit pto.fusion_region results / pto.yield operands, while
// external inputs stay implicitly captured and scratch alloc_tile temporaries
// can be sunk into the region body. This also makes pto.yield the explicit
// summary of which internal tiles still escape the region boundary.

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

// LIB-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// LIB-LABEL: func.func @fusion_region_interface(
// LIB: %[[REGION:.*]]:2 = pto.fusion_region {
// LIB: %[[TMP0:[0-9]+]] = pto.bind_tile
// LIB: %[[TMP1:[0-9]+]] = pto.bind_tile
// LIB: %[[SUM:[0-9]+]] = pto.bind_tile
// LIB: func.call @__pto_oplib_inst_l3_broadcast_row_binary_template_trowexpandmul_linear(%arg0, %arg2, %[[TMP0]])
// LIB: func.call @__pto_oplib_inst_l3_broadcast_row_binary_template_trowexpandmul_linear(%arg1, %arg3, %[[TMP1]])
// LIB: func.call @__pto_oplib_inst_l3_float_binary_elementwise_template_tadd_tile(%[[TMP0]], %[[TMP1]], %[[SUM]])
// LIB: pto.yield(%[[TMP0]], %[[SUM]]) : (memref
// LIB: } {pto.fusion.group_id = 0 : i64} : memref
// LIB: pto.ttrans ins(%[[REGION]]#0, %{{.*}})
// LIB: pto.ttrans ins(%[[REGION]]#1, %{{.*}})
// LIB: return

// INLINE-LABEL: IR Dump After PTOInlineLibCall
// INLINE-LABEL: func.func @fusion_region_interface(
// INLINE: %[[REGION:.*]]:2 = pto.fusion_region {
// INLINE: %[[TMP0:[0-9]+]] = pto.bind_tile
// INLINE: %[[TMP1:[0-9]+]] = pto.bind_tile
// INLINE: %[[SUM:[0-9]+]] = pto.bind_tile
// INLINE: pto.simd.vec_scope {
// INLINE: pto.simd.vec_scope {
// INLINE: pto.simd.vec_scope {
// INLINE: pto.yield(%[[TMP0]], %[[SUM]]) : (memref
// INLINE: } {pto.fusion.group_id = 0 : i64} : memref
// INLINE: pto.ttrans ins(%[[REGION]]#0, %{{.*}})
// INLINE: pto.ttrans ins(%[[REGION]]#1, %{{.*}})
// INLINE: return

// LOW-LABEL: IR Dump After PTOLowLevelLoopFusion
// LOW-LABEL: func.func @fusion_region_interface(
// LOW: %[[REGION:.*]]:2 = pto.fusion_region {
// LOW: %[[TMP0:[0-9]+]] = pto.bind_tile
// LOW: %[[TMP1:[0-9]+]] = pto.bind_tile
// LOW: %[[SUM:[0-9]+]] = pto.bind_tile
// LOW: pto.simd.vec_scope {
// LOW: pto.simd.vec_scope {
// LOW: pto.simd.vec_scope {
// LOW: pto.yield(%[[TMP0]], %[[SUM]]) : (memref
// LOW: } {pto.fusion.group_id = 0 : i64} : memref
// LOW: pto.ttrans ins(%[[REGION]]#0, %{{.*}})
// LOW: pto.ttrans ins(%[[REGION]]#1, %{{.*}})
// LOW: return

// ELIDE-LABEL: IR Dump After PTOFusionLoadStoreElision
// ELIDE-LABEL: func.func @fusion_region_interface(
// ELIDE: %[[REGION:.*]]:2 = pto.fusion_region {
// ELIDE: %[[TMP0:[0-9]+]] = pto.bind_tile
// ELIDE: %[[TMP1:[0-9]+]] = pto.bind_tile
// ELIDE: %[[SUM:[0-9]+]] = pto.bind_tile
// ELIDE: pto.yield(%[[TMP0]], %[[SUM]]) : (memref
// ELIDE: } {pto.fusion.group_id = 0 : i64} : memref
// ELIDE: pto.ttrans ins(%[[REGION]]#0, %{{.*}})
// ELIDE: pto.ttrans ins(%[[REGION]]#1, %{{.*}})
// ELIDE: return
