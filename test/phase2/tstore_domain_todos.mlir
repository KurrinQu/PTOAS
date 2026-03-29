// RUN: ./build/tools/ptoas/ptoas --pto-backend=vpto --emit-vpto %s -o - 2>/dev/null | FileCheck %s

// CHECK-LABEL: func.func @tstore_domain_todos
// CHECK: TSTORE ACC lowering TODO for vpto backend
// CHECK: TSTORE MAT lowering TODO for vpto backend
// CHECK-NOT: failed to legalize operation

module {
  func.func @tstore_domain_todos(%dst: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %tv = pto.make_tensor_view %dst, shape = [%c32, %c32], strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>
    %slice = pto.partition_view %tv, offsets = [%c0, %c0], sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %vec = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %acc = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    %mat = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    pto.tstore ins(%vec : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      outs(%slice : !pto.partition_tensor_view<32x32xf32>)
    pto.tstore ins(%acc : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
      outs(%slice : !pto.partition_tensor_view<32x32xf32>)
    pto.tstore ins(%mat : !pto.tile_buf<loc=mat, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
      outs(%slice : !pto.partition_tensor_view<32x32xf32>)
    return
  }
}
