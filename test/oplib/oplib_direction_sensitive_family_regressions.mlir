// RUN: rm -rf %t.dir && mkdir -p %t.dir/families
// RUN: cp %S/../../oplib/level3/float_tile_scalar_templates.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/broadcast_row_binary_templates.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/families/a5_oplib_v1_manifest.yaml %t.dir/families/
// RUN: { ptoas %s --enable-op-fusion --op-lib-dir=%t.dir --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s
// XFAIL: *

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_float_tile_scalar_template_tdivs_tile_scalar(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_float_tile_scalar_template_tdivs_scalar_tile(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_broadcast_row_binary_template_trowexpanddiv_linear(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_broadcast_row_binary_template_trowexpandsub_linear(
// CHECK-DAG: pto.oplib.instance.op = "tdivs"
// CHECK-DAG: pto.oplib.instance.op = "trowexpanddiv"
// CHECK-DAG: pto.oplib.instance.op = "trowexpandsub"
// CHECK-DAG: pto.oplib.instance.variant_id = "tile_scalar"
// CHECK-DAG: pto.oplib.instance.variant_id = "scalar_tile"
// CHECK-DAG: pto.oplib.instance.variant_id = "linear"
// CHECK: call @__pto_oplib_inst_l3_float_tile_scalar_template_tdivs_tile_scalar(
// CHECK: call @__pto_oplib_inst_l3_float_tile_scalar_template_tdivs_scalar_tile(
// CHECK: call @__pto_oplib_inst_l3_broadcast_row_binary_template_trowexpanddiv_linear(
// CHECK: call @__pto_oplib_inst_l3_broadcast_row_binary_template_trowexpanddiv_linear(
// CHECK: call @__pto_oplib_inst_l3_broadcast_row_binary_template_trowexpandsub_linear(
// CHECK: call @__pto_oplib_inst_l3_broadcast_row_binary_template_trowexpandsub_linear(

module {
  func.func @direction_sensitive_family_regressions() {
    %cst = arith.constant 3.000000e+00 : f32

    %full = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %full1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %row = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    %divTileScalar = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %divScalarTile = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %rowDiv0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %rowDiv1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %rowSub0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %rowSub1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tdivs ins(%full, %cst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f32) outs(%divTileScalar : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tdivs ins(%cst, %full : f32, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%divScalarTile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    pto.trowexpanddiv ins(%full, %row : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%rowDiv0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.trowexpanddiv ins(%row, %full : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%rowDiv1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    pto.trowexpandsub ins(%full1, %row : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%rowSub0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.trowexpandsub ins(%row, %full1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%rowSub1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
