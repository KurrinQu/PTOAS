// RUN: { ptoas %s --pto-arch=a5 --pto-level=level3 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all -o /dev/null 2>&1 || true; } | FileCheck %s
// XFAIL: *

module {
  // CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
  // CHECK-DAG: pto.oplib.instance.kind = "l3_float_tile_scalar_template"
  // CHECK-DAG: pto.oplib.instance.op = "tadds"
  // CHECK-DAG: pto.oplib.instance.variant_id = "tile_scalar_f16"
  // CHECK-DAG: pto.oplib.instance.kind = "l3_int_binary_elementwise_template"
  // CHECK-DAG: pto.oplib.instance.op = "tadd"
  // CHECK-DAG: pto.oplib.instance.variant_id = "tile"
  // CHECK-DAG: pto.oplib.instance.kind = "l3_int_tile_scalar_elementwise_template"
  // CHECK-DAG: pto.oplib.instance.op = "tdivs"
  // CHECK-DAG: pto.oplib.instance.variant_id = "scalar_tile"
  // CHECK-DAG: pto.oplib.instance.kind = "l3_int_unary_template"
  // CHECK-DAG: pto.oplib.instance.op = "tneg"
  // CHECK-DAG: pto.oplib.instance.op = "trelu"
  func.func @arith_family_int_paths() {
    %c0_i64 = arith.constant 0 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %f16_one = arith.constant 1.0 : f16
    %i32_seven = arith.constant 7 : i32

    %f16_src = pto.alloc_tile addr = %c0_i64 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f16_dst = pto.alloc_tile addr = %c4096_i64 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %i32_src0 = pto.alloc_tile addr = %c0_i64 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %i32_src1 = pto.alloc_tile addr = %c4096_i64 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %i32_tmp = pto.alloc_tile addr = %c8192_i64 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %i32_dst = pto.alloc_tile addr = %c12288_i64 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tadds ins(%f16_src, %f16_one : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f16) outs(%f16_dst : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tadd ins(%i32_src0, %i32_src1 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%i32_tmp : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tdivs ins(%i32_seven, %i32_tmp : i32, !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%i32_tmp : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tneg ins(%i32_tmp : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%i32_src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.trelu ins(%i32_src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%i32_dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
