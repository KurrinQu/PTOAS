// RUN: { ptoas %s --pto-arch=a5 --pto-level=level3 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all -o /dev/null 2>&1 || true; } | FileCheck %s
// XFAIL: *

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-LABEL: func.func @key_family_dtype_lowering_smoke()
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_float_unary_template_trecip_tile_f16(
// CHECK-DAG: pto.oplib.instance.dtype = "f16"
// CHECK-DAG: pto.oplib.instance.op = "trecip"
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_cmp_tile_tile_template_tcmp_ge(
// CHECK-DAG: pto.oplib.instance.op = "tcmp"
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_cmp_tile_scalar_template_tcmps_ge(
// CHECK-DAG: pto.oplib.instance.op = "tcmps"
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_select_mask_template_tsel_mask(
// CHECK-DAG: pto.oplib.instance.op = "tsel"
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_reduce_row_template_trowsum_linear(
// CHECK-DAG: pto.oplib.instance.op = "trowsum"
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_reduce_colsum_template_tcolsum_linear(
// CHECK-DAG: pto.oplib.instance.op = "tcolsum"
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_broadcast_row_template_trowexpand_linear(
// CHECK-DAG: pto.oplib.instance.op = "trowexpand"
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_broadcast_col_template_tcolexpand_linear(
// CHECK-DAG: pto.oplib.instance.op = "tcolexpand"
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_scalar_expand_template_texpands_scalar(
// CHECK-DAG: pto.oplib.instance.op = "texpands"
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_int_binary_elementwise_template_tand_tile_i16(
// CHECK-DAG: pto.oplib.instance.dtype = "i16"
// CHECK-DAG: pto.oplib.instance.op = "tand"
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_int_unary_template_tnot_tile_i16(
// CHECK-DAG: pto.oplib.instance.op = "tnot"
// CHECK-DAG: call @__pto_oplib_inst_l3_float_unary_template_trecip_tile_f16(
// CHECK-DAG: call @__pto_oplib_inst_l3_cmp_tile_tile_template_tcmp_ge(
// CHECK-DAG: call @__pto_oplib_inst_l3_cmp_tile_scalar_template_tcmps_ge(
// CHECK-DAG: call @__pto_oplib_inst_l3_select_mask_template_tsel_mask(
// CHECK-DAG: call @__pto_oplib_inst_l3_reduce_row_template_trowsum_linear(
// CHECK-DAG: call @__pto_oplib_inst_l3_reduce_colsum_template_tcolsum_linear(
// CHECK-DAG: call @__pto_oplib_inst_l3_broadcast_row_template_trowexpand_linear(
// CHECK-DAG: call @__pto_oplib_inst_l3_broadcast_col_template_tcolexpand_linear(
// CHECK-DAG: call @__pto_oplib_inst_l3_scalar_expand_template_texpands_scalar(
// CHECK-DAG: call @__pto_oplib_inst_l3_int_binary_elementwise_template_tand_tile_i16(
// CHECK-DAG: call @__pto_oplib_inst_l3_int_unary_template_tnot_tile_i16(

module {
  func.func @key_family_dtype_lowering_smoke() {
    %f16_scale = arith.constant 1.000000e+00 : f16
    %c0_i64 = arith.constant 0 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c20480_i64 = arith.constant 20480 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c28672_i64 = arith.constant 28672 : i64
    %c32768_i64 = arith.constant 32768 : i64

    %f16_a = pto.alloc_tile addr = %c0_i64 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f16_b = pto.alloc_tile addr = %c4096_i64 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f16_c = pto.alloc_tile addr = %c8192_i64 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %row = pto.alloc_tile addr = %c12288_i64 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %col = pto.alloc_tile addr = %c16384_i64 : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32, v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %i16_a = pto.alloc_tile addr = %c20480_i64 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %i16_b = pto.alloc_tile addr = %c24576_i64 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %mask = pto.alloc_tile addr = %c28672_i64 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f16_d = pto.alloc_tile addr = %c32768_i64 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.trecip ins(%f16_a : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f16_b : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tcmp ins(%f16_a, %f16_b {cmpMode = #pto<cmp ge>} : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%mask : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tcmps ins(%f16_a, %f16_scale {cmpMode = #pto<cmp ge>} : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f16) outs(%mask : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tsel ins(%mask, %f16_a, %f16_b, %f16_d : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f16_c : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.trowsum ins(%f16_a, %f16_b : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%row : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tcolsum ins(%f16_a, %f16_b {isBinary = false} : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%col : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32, v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.trowexpand ins(%row : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f16_a : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tcolexpand ins(%f16_a : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f16_b : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.texpands ins(%f16_scale : f16) outs(%f16_b : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tand ins(%i16_a, %i16_b : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%i16_a : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tnot ins(%i16_a : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%i16_b : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
