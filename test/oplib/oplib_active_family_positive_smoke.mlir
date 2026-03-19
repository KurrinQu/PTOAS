// RUN: ptoas %s --pto-arch=a5 --pto-level=level3 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=active_family_positive_smoke -o /dev/null > %t.out 2>&1 || true
// RUN: FileCheck %s < %t.out
// XFAIL: *

module {
  // CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
  // CHECK-LABEL: func.func @active_family_positive_smoke()
  // CHECK-DAG: call @__pto_oplib_inst_l3_float_binary_elementwise_template_tadd{{.*}}(
  // CHECK-DAG: call @__pto_oplib_inst_l3_int_binary_elementwise_template_tand{{.*}}(
  // CHECK-DAG: call @__pto_oplib_inst_l3_float_partial_binary_template_tpartadd{{.*}}(
  // CHECK-DAG: call @__pto_oplib_inst_l3_float_tile_scalar_template_tadds{{.*}}(
  // CHECK-DAG: call @__pto_oplib_inst_l3_int_tile_scalar_elementwise_template_tdivs{{.*}}(
  // CHECK-DAG: call @__pto_oplib_inst_l3_float_unary_template_trelu{{.*}}(
  // CHECK-DAG: call @__pto_oplib_inst_l3_float_unary_template_trecip{{.*}}(
  // CHECK-DAG: call @__pto_oplib_inst_l3_float_unary_math_template_texp{{.*}}(
  // CHECK-DAG: call @__pto_oplib_inst_l3_int_unary_template_tnot{{.*}}(
  // CHECK-DAG: call @__pto_oplib_inst_l3_reduce_row_template_trowsum_linear(
  // CHECK-DAG: call @__pto_oplib_inst_l3_reduce_col_template_tcolmax_linear(
  // CHECK-DAG: call @__pto_oplib_inst_l3_reduce_colsum_template_tcolsum_linear(
  // CHECK-DAG: call @__pto_oplib_inst_l3_broadcast_row_template_trowexpand_linear(
  // CHECK-DAG: call @__pto_oplib_inst_l3_broadcast_col_template_tcolexpand_linear(
  // CHECK-DAG: call @__pto_oplib_inst_l3_broadcast_row_binary_template_trowexpandmul_linear(
  // CHECK-DAG: call @__pto_oplib_inst_l3_scalar_expand_template_texpands_scalar(
  // CHECK-DAG: call @__pto_oplib_inst_l3_cmp_tile_tile_template_tcmp_lt(
  // CHECK-DAG: call @__pto_oplib_inst_l3_cmp_tile_scalar_template_tcmps_ge(
  // CHECK-DAG: call @__pto_oplib_inst_l3_select_mask_template_tsel_mask(
  func.func @active_family_positive_smoke() {
    %f16_scale = arith.constant 1.000000e+00 : f16
    %f32_scale = arith.constant 1.250000e+00 : f32
    %i32_shift = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c20480_i64 = arith.constant 20480 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c28672_i64 = arith.constant 28672 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c36864_i64 = arith.constant 36864 : i64
    %c40960_i64 = arith.constant 40960 : i64
    %c45056_i64 = arith.constant 45056 : i64

    %f32_a = pto.alloc_tile addr = %c0_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f32_b = pto.alloc_tile addr = %c4096_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f32_c = pto.alloc_tile addr = %c8192_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f16_a = pto.alloc_tile addr = %c12288_i64 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f16_b = pto.alloc_tile addr = %c16384_i64 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %i32_a = pto.alloc_tile addr = %c20480_i64 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %i32_b = pto.alloc_tile addr = %c24576_i64 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %i32_c = pto.alloc_tile addr = %c28672_i64 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %mask = pto.alloc_tile addr = %c32768_i64 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %row = pto.alloc_tile addr = %c36864_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %col = pto.alloc_tile addr = %c40960_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f32_tmp = pto.alloc_tile addr = %c45056_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tadd ins(%f32_a, %f32_b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f32_c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tand ins(%i32_a, %i32_b : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%i32_c : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tpartadd ins(%f32_a, %f32_b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f32_c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tadds ins(%f16_a, %f16_scale : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f16) outs(%f16_b : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tdivs ins(%i32_a, %i32_shift : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, i32) outs(%i32_c : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.trelu ins(%f32_c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f32_a : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.trecip ins(%f32_a : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f32_c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.texp ins(%f32_a : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f32_b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tnot ins(%i32_c : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%i32_a : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    pto.trowsum ins(%f32_a, %f32_b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%row : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tcolmax ins(%f32_a : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f32_c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tcolsum ins(%f32_a, %f32_b {isBinary = false} : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%col : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.trowexpand ins(%row : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f32_a : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tcolexpand ins(%f32_a : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f32_b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.trowexpandmul ins(%f32_b, %row : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=32, v_col=1, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f32_c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.texpands ins(%f32_scale : f32) outs(%f32_a : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    pto.tcmp ins(%f32_b, %f32_c {cmpMode = #pto<cmp lt>} : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%mask : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tcmps ins(%f32_b, %f32_scale {cmpMode = #pto<cmp ge>} : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f32) outs(%mask : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tsel ins(%mask, %f32_b, %f32_c, %f32_tmp : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f32_a : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
