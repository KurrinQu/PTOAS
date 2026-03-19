// RUN: { ptoas %s --pto-arch=a5 --pto-level=level3 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all -o /dev/null 2>&1 || true; } | FileCheck %s
// XFAIL: *

module {
  // CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
  // CHECK-DAG: pto.oplib.instance.kind = "l3_float_partial_binary_template"
  // CHECK-DAG: pto.oplib.instance.op = "tpartadd"
  // CHECK-DAG: pto.oplib.instance.variant_id = "tile_f32"
  // CHECK-DAG: pto.oplib.instance.kind = "l3_float_partial_binary_template"
  // CHECK-DAG: pto.oplib.instance.op = "tpartmax"
  // CHECK-DAG: pto.oplib.instance.variant_id = "tile_i32"
  // CHECK-DAG: pto.oplib.instance.kind = "l3_float_partial_binary_template"
  // CHECK-DAG: pto.oplib.instance.op = "tprelu"
  // CHECK-DAG: pto.oplib.instance.variant_id = "tile_f16"
  // CHECK-DAG: pto.oplib.instance.kind = "l3_float_tile_scalar_template"
  // CHECK-DAG: pto.oplib.instance.op = "tlrelu"
  func.func @partial_binary_activation_paths() {
    %c0_i64 = arith.constant 0 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c20480_i64 = arith.constant 20480 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c28672_i64 = arith.constant 28672 : i64
    %f32_slope = arith.constant 1.000000e-01 : f32

    %f32_src0 = pto.alloc_tile addr = %c0_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f32_src1 = pto.alloc_tile addr = %c4096_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f32_dst = pto.alloc_tile addr = %c8192_i64 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %i32_src0 = pto.alloc_tile addr = %c12288_i64 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %i32_src1 = pto.alloc_tile addr = %c16384_i64 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %i32_dst = pto.alloc_tile addr = %c20480_i64 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f16_src = pto.alloc_tile addr = %c24576_i64 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %f16_slopes = pto.alloc_tile addr = %c28672_i64 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %tmp = pto.alloc_tile addr = %c0_i64 : !pto.tile_buf<loc=vec, dtype=ui8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tpartadd ins(%f32_src0, %f32_src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f32_dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tpartmax ins(%i32_src0, %i32_src1 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%i32_dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tprelu ins(%f16_src, %f16_slopes, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=ui8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%f16_src : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tlrelu ins(%f32_dst, %f32_slope : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f32) outs(%f32_dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
