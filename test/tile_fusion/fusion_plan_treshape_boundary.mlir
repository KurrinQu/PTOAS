// RUN: { ptoas %s --enable-op-fusion --pto-backend=a5vm --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=treshape_boundary -o /dev/null 2>&1 || true; } | awk '/IR Dump After FusionPlan/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /FusionPlan/) exit; print}' | FileCheck %s

module {
  func.func @treshape_boundary(
      %arg0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %arg1: !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %arg2: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {
    %tmp0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %tmp1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %tmp2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %tmp3 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tadd ins(%arg0, %arg0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    %view = pto.treshape %tmp0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tmul ins(%view, %arg1 : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp1 : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tadd ins(%arg2, %arg2 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp2 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tmul ins(%tmp2, %arg2 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp3 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}

// CHECK-LABEL: IR Dump After FusionPlan
// CHECK-LABEL: func.func @treshape_boundary(
// CHECK: pto.tadd ins(%arg0, %arg0
// CHECK-NOT: pto.fusion.group_id
// CHECK: %[[RESHAPED:[0-9]+]] = pto.treshape
// CHECK-NOT: pto.fusion.group_id
// CHECK: pto.tmul ins(%[[RESHAPED]], %arg1
// CHECK-NOT: pto.fusion.group_id
// CHECK: pto.tadd ins(%arg2, %arg2
// CHECK-SAME: outs(%[[CHAIN_DST:[0-9]+]]
// CHECK-SAME: pto.fusion.group_id = 0 : i64
// CHECK-SAME: pto.fusion.order = 0 : i64
// CHECK: pto.tmul ins(%[[CHAIN_DST]], %arg2
// CHECK-SAME: pto.fusion.group_id = 0 : i64
// CHECK-SAME: pto.fusion.order = 1 : i64
