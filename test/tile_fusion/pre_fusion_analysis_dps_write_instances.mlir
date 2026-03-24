// RUN: ptoas %s --dump-pre-fusion-analysis 2>&1 | FileCheck %s

module {
  func.func @dps_reused_destination(
      %in0: !pto.ptr<f32>,
      %out0: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index

    %in_tv = pto.make_tensor_view %in0, shape = [%c32, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %out_tv = pto.make_tensor_view %out0, shape = [%c32, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %in_pt = pto.partition_view %in_tv, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %out_pt = pto.partition_view %out_tv, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

    %tmp = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tload ins(%in_pt : !pto.partition_tensor_view<32x32xf32>) outs(%tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tadd ins(%tmp, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tmul ins(%tmp, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%out_pt : !pto.partition_tensor_view<32x32xf32>)
    return
  }
}

// CHECK-LABEL: PreFusionAnalysis @dps_reused_destination
// CHECK: block[0]
// CHECK: domain_class[0] domain=(32x32) proof=proven reason=none members=[0, 1]
// CHECK: compute[0] op=tadd family=elementwise domain_class=0 inputs=[node0.out0, node0.out0] outputs=[node0.out0] incoming=[0, 1] outgoing=[]
// CHECK: compute[1] op=tmul family=elementwise domain_class=0 inputs=[node0.out0, node0.out0] outputs=[node0.out0] incoming=[2, 3] outgoing=[]
// CHECK: edge[0] producer=0 consumer=0 value=node0.out0
// CHECK: edge[1] producer=0 consumer=0 value=node0.out0
// CHECK: edge[2] producer=1 consumer=1 value=node0.out0
// CHECK: edge[3] producer=1 consumer=1 value=node0.out0
// CHECK: liveness value=node0.out0 producer=1 consumers=[0, 1] write_instances=[0, 1]
// CHECK-SAME: last_local_consumer=0
// CHECK-SAME: hard_boundary_users=true
// CHECK: write_instance[0] value=node0.out0 storage=node0.out0 producer=0 consumers=[1] last_local_consumer=1 escape_class=internal
// CHECK: write_instance[1] value=node0.out0 storage=node0.out0 producer=1 consumers=[] last_local_consumer=<none> escape_class=hard_external
