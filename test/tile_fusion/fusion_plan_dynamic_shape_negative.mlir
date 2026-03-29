// RUN: { ptoas %s --enable-op-fusion --pto-backend=vpto --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=fusion_plan_dynamic_shape_negative -o /dev/null 2>&1; } | FileCheck %s

module {
  func.func @fusion_plan_dynamic_shape_negative(
      %arg0: !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %vrow: index,
      %vcol: index) {
    %tmp0 = pto.alloc_tile valid_row = %vrow valid_col = %vcol : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %tmp1 = pto.alloc_tile valid_row = %vrow valid_col = %vcol : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tadd ins(%arg0, %arg0 : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp0 : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tmul ins(%tmp0, %arg0 : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp1 : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}

// CHECK-LABEL: IR Dump After FusionPlan
// CHECK-LABEL: func.func @fusion_plan_dynamic_shape_negative(
// CHECK: pto.tadd ins(%arg0, %arg0
// CHECK-NOT: pto.fusion.group_id
// CHECK: pto.tmul ins(%0, %arg0
// CHECK-NOT: pto.fusion.group_id
// CHECK: return
