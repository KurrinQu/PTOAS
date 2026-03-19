// RUN: FileCheck %s --check-prefix=TEMPLATE < %S/../../oplib/level3/float_unary_templates.mlir
// RUN: ptoas %s --pto-arch=a5 --pto-level=level3 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=trecip_public_api_rewrite -o /dev/null > %t.out 2>&1 || true
// RUN: FileCheck %s --check-prefix=LOWERING < %t.out
// XFAIL: *

// TEMPLATE-LABEL: func.func private @__pto_oplib_variant_trecip_f16(
// TEMPLATE: pto.oplib.op = "trecip"
// TEMPLATE: pto.oplib.semantic_origin = "public_api_rewrite"
// TEMPLATE: pto.oplib.semantic_equivalent = "TDIVS(dst, 1, src)"
// TEMPLATE: %ones = arith.constant dense<1.0> : vector<128xf16>
// TEMPLATE: %result = arith.divf %ones, %lhs
// TEMPLATE-LABEL: func.func private @__pto_oplib_variant_trecip_f32(
// TEMPLATE: pto.oplib.semantic_origin = "public_api_rewrite"
// TEMPLATE: pto.oplib.semantic_equivalent = "TDIVS(dst, 1, src)"

module {
  // LOWERING-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
  // LOWERING-LABEL: func.func @trecip_public_api_rewrite()
  // LOWERING-DAG: call @__pto_oplib_inst_l3_float_unary_template_trecip_tile_f16(
  // LOWERING-DAG: call @__pto_oplib_inst_l3_float_unary_template_trecip_tile(
  func.func @trecip_public_api_rewrite() {
    %c0_i64 = arith.constant 0 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c12288_i64 = arith.constant 12288 : i64

    %f16_src = pto.alloc_tile addr = %c0_i64 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f16_dst = pto.alloc_tile addr = %c4096_i64 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f32_src = pto.alloc_tile addr = %c8192_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f32_dst = pto.alloc_tile addr = %c12288_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.trecip ins(%f16_src : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f16_dst : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.trecip ins(%f32_src : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f32_dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
