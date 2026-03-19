// RUN: rm -rf %t.dir && mkdir -p %t.dir/families
// RUN: cp %S/../../oplib/level3/reduce_colsum_templates.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/families/a5_oplib_v1_manifest.yaml %t.dir/families/
// RUN: ptoas %s --enable-op-fusion --op-lib-dir=%t.dir --pto-arch=a5 -o %t.cpp
// RUN: FileCheck %s --check-prefix=CPP < %t.cpp

// CPP: bool {{.*}} = true;
// CPP: TCOLSUM(

module {
  func.func @allow_reduce_colsum_binary_variant() {
    %src = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %tmp = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %dst = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tcolsum ins(%src, %tmp {isBinary = true} : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
