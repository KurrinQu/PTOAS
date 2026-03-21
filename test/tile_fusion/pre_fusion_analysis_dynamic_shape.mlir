// RUN: ptoas %s --dump-pre-fusion-analysis 2>&1 | FileCheck %s

module {
  func.func @dynamic_chain(
      %arg0: !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %vrow: index,
      %vcol: index) {
    %tmp0 = pto.alloc_tile valid_row = %vrow valid_col = %vcol : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %tmp1 = pto.alloc_tile valid_row = %vrow valid_col = %vcol : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tadd ins(%arg0, %arg0 : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp0 : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tmul ins(%tmp0, %arg0 : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tmp1 : !pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}

// CHECK-LABEL: PreFusionAnalysis @dynamic_chain
// CHECK: block[0]
// CHECK: domain_class[0] domain=(?x?) proof=unproven reason=dynamic_shape members=[0]
// CHECK: domain_class[1] domain=(?x?) proof=unproven reason=dynamic_shape members=[1]
// CHECK: compute[0] op=tadd family=elementwise domain_class=0
// CHECK: compute[1] op=tmul family=elementwise domain_class=1
// CHECK: edge[0] producer=0 consumer=1 value=node0.out0
