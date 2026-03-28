// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: mkdir -p %t.dir/families
// RUN: cp %S/../../oplib/level3/float_tile_scalar_templates.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/cmp_tile_scalar_templates.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/select_mask_templates.mlir %t.dir/
// RUN: cp %S/resources/good_reduce_colsum_templates.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/broadcast_row_binary_templates.mlir %t.dir/
// XFAIL: *
// RUN: cp %S/../../oplib/level3/families/a5_oplib_v1_manifest.yaml %t.dir/families/
// RUN: { ptoas %s --enable-op-fusion --op-lib-dir=%t.dir --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_float_tile_scalar_template_tdivs_tile_scalar(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_float_tile_scalar_template_tdivs_scalar_tile(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_cmp_tile_scalar_template_tcmps_lt(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_cmp_tile_scalar_template_tcmps_ge(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_select_mask_template_tsel_mask(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_reduce_colsum_template_tcolsum_linear(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_broadcast_row_binary_template_trowexpandmul_linear(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_broadcast_row_binary_template_trowexpandmul_linear_1(
// CHECK-DAG: pto.oplib.instance.kind = "l3_float_tile_scalar_template"
// CHECK-DAG: pto.oplib.instance.kind = "l3_cmp_tile_scalar_template"
// CHECK-DAG: pto.oplib.instance.kind = "l3_select_mask_template"
// CHECK-DAG: pto.oplib.instance.kind = "l3_reduce_colsum_template"
// CHECK-DAG: pto.oplib.instance.kind = "l3_broadcast_row_binary_template"
// CHECK-DAG: pto.oplib.instance.op = "tdivs"
// CHECK-DAG: pto.oplib.instance.op = "tcmps"
// CHECK-DAG: pto.oplib.instance.op = "tsel"
// CHECK-DAG: pto.oplib.instance.op = "tcolsum"
// CHECK-DAG: pto.oplib.instance.op = "trowexpandmul"
// CHECK-DAG: pto.oplib.instance.dtype = "f32"
// CHECK-DAG: pto.oplib.instance.variant_id = "tile_scalar"
// CHECK-DAG: pto.oplib.instance.variant_id = "scalar_tile"
// CHECK-DAG: pto.oplib.instance.variant_id = "lt"
// CHECK-DAG: pto.oplib.instance.variant_id = "ge"
// CHECK-DAG: pto.oplib.instance.variant_id = "mask"
// CHECK-DAG: pto.oplib.instance.variant_id = "linear"
// CHECK: call @__pto_oplib_inst_l3_float_tile_scalar_template_tdivs_tile_scalar(
// CHECK: call @__pto_oplib_inst_l3_float_tile_scalar_template_tdivs_scalar_tile(
// CHECK: call @__pto_oplib_inst_l3_cmp_tile_scalar_template_tcmps_lt(
// CHECK: call @__pto_oplib_inst_l3_cmp_tile_scalar_template_tcmps_ge(
// CHECK: call @__pto_oplib_inst_l3_select_mask_template_tsel_mask(
// CHECK: call @__pto_oplib_inst_l3_reduce_colsum_template_tcolsum_linear(
// CHECK: call @__pto_oplib_inst_l3_broadcast_row_binary_template_trowexpandmul_linear(
// CHECK: call @__pto_oplib_inst_l3_broadcast_row_binary_template_trowexpandmul_linear(
// CHECK: call @__pto_oplib_inst_l3_broadcast_row_binary_template_trowexpandmul_linear_1(

module {
  func.func @instance_selection_compat() {
    %cst = arith.constant 2.000000e+00 : f32

    %src0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %src1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %tmp = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %row = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %dst0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %dst1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %cmpDst = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %maskDst = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %colLinear = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %rowDst0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %rowDst1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %rowDst2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tdivs ins(%src0, %cst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f32) outs(%dst0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tdivs ins(%cst, %src0 : f32, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%dst1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    pto.tcmps ins(%src0, %cst {cmpMode = #pto<cmp lt>} : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f32) outs(%cmpDst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tcmps ins(%src0, %cst {cmpMode = #pto<cmp ge>} : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f32) outs(%cmpDst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    pto.tsel ins(%cmpDst, %src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%maskDst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    pto.tcolsum ins(%src0, %tmp {isBinary = false} : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%colLinear : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.trowexpandmul ins(%src0, %row : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%rowDst0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.trowexpandmul ins(%row, %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%rowDst1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.trowexpandmul ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%rowDst2 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
