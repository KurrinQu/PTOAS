// RUN: ptoas %s --dump-pre-fusion-analysis 2>&1 | FileCheck %s

module {
  func.func @treshape_boundary(
      %arg0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %arg1: !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %arg2: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {
    %tmp0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %tmp1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %tmp2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %tmp3 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tadd ins(%arg0, %arg0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    %view = pto.treshape %tmp0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tmul ins(%view, %arg1 : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp1 : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=16, v_row=64, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tadd ins(%arg2, %arg2 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp2 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tmul ins(%tmp2, %arg2 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp3 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}

// CHECK-LABEL: PreFusionAnalysis @treshape_boundary
// CHECK: block[0]
// CHECK: local_boundary[0] op=treshape inputs=[node0.out0] outputs=[boundary#0]
// CHECK: compute[0] op=tadd family=elementwise domain_class=0 inputs=[external#0, external#0] outputs=[node0.out0] incoming=[] outgoing=[]
// CHECK: compute[1] op=tmul family=elementwise domain_class=1 inputs=[boundary#0, external#1] outputs=[node1.out0] incoming=[] outgoing=[]
// CHECK: compute[2] op=tadd family=elementwise domain_class=0 inputs=[external#2, external#2] outputs=[node2.out0] incoming=[] outgoing=[0]
// CHECK: compute[3] op=tmul family=elementwise domain_class=0 inputs=[node2.out0, external#2] outputs=[node3.out0] incoming=[0] outgoing=[]
// CHECK: edge[0] producer=2 consumer=3 value=node2.out0
// CHECK: liveness value=node0.out0 producer=0 consumers=[0] write_instances=[0]
// CHECK-SAME: boundary_users=true
// CHECK: write_instance[0] value=node0.out0 storage=node0.out0 producer=0 consumers=[] last_local_consumer=<none> escape_class=local_boundary_external
