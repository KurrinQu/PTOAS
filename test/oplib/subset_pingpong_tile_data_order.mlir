// RUN: ptoas %s --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.cpp
// RUN: FileCheck %s < %t.cpp

module {
  func.func @test_double_buffer_step(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=64, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %1 = pto.make_tensor_view %arg0, shape = [%c32, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %2 = pto.make_tensor_view %arg1, shape = [%c32, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %3 = pto.partition_view %1, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %4 = pto.partition_view %2, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %5 = pto.subset %0[%c0, %c0] sizes [32, 32] : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=64, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %6 = pto.subset %0[%c0, %c32] sizes [32, 32] : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=64, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%3 : !pto.partition_tensor_view<32x32xf32>) outs(%6 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tadd ins(%6, %6 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%5 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%5 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%4 : !pto.partition_tensor_view<32x32xf32>)
    return
  }
}

// The subset tiles are created via TASSIGN(tile, addr). Make sure we only call
// tile.data() after that assignment; otherwise it reads the uninitialized tile
// address and results become nondeterministic on device.
//
// CHECK: Tile<{{.*}}, float, 32, 32, {{.*}}> [[PING:v[0-9]+]];
// CHECK-NOT: [[PING]].data()
// CHECK: TASSIGN([[PING]],
// CHECK: [[PING]].data()
//
// CHECK: Tile<{{.*}}, float, 32, 32, {{.*}}> [[PONG:v[0-9]+]];
// CHECK-NOT: [[PONG]].data()
// CHECK: TASSIGN([[PONG]],
// CHECK: [[PONG]].data()
