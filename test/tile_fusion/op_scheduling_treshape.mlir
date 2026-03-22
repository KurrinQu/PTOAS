// RUN: { ptoas %s --enable-op-fusion --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=op_scheduling_treshape -o /dev/null 2>&1 || true; } | FileCheck %s

// 5.4 treshape scheduling regression:
// an unrelated pto.treshape sits between members of one planned fusion group.
// OpSchedulingPass must compact the group into a contiguous span by moving
// later group members across that local boundary, while keeping the unrelated
// treshape chain outside the fused span.

module {
  func.func @op_scheduling_treshape(
      %full0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %full1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %full2: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %full3: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %wide: !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %row0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %row1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {
    %tmp0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %noise0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %tmp1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %tmp2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %noise1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.trowexpandmul ins(%full0, %row0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tadd ins(%full2, %full3 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%noise0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    %view = pto.treshape %noise0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.trowexpandmul ins(%full1, %row1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tadd ins(%tmp0, %tmp1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp2 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tmul ins(%view, %wide : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%noise1 : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}

// CHECK-LABEL: IR Dump After OpScheduling
// CHECK-LABEL: func.func @op_scheduling_treshape(
// CHECK: pto.trowexpandmul{{.*}}outs(%[[DST0:[0-9]+]]
// CHECK-SAME: pto.fusion.group_id = 0 : i64
// CHECK-SAME: pto.fusion.order = 0 : i64
// CHECK-NEXT: pto.trowexpandmul{{.*}}outs(%[[DST1:[0-9]+]]
// CHECK-SAME: pto.fusion.group_id = 0 : i64
// CHECK-SAME: pto.fusion.order = 1 : i64
// CHECK-NEXT: pto.tadd ins(%[[DST0]], %[[DST1]]
// CHECK-SAME: pto.fusion.group_id = 0 : i64
// CHECK-SAME: pto.fusion.order = 2 : i64
// CHECK-NEXT: pto.tadd ins(%arg2, %arg3
// CHECK-NEXT: %[[RESHAPED:[0-9]+]] = pto.treshape
// CHECK-NEXT: pto.tmul ins(%[[RESHAPED]], %arg4
