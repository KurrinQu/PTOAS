// RUN: ptoas --enable-cv-separation %s | FileCheck %s

// Test: scf.for containing both cube and vector ops is split into
// two parallel loops, one in each section.

module {
  func.func @mixed_loop(
      %gm_a: memref<16x256xf16, #pto.address_space<gm>>,
      %mat_a: memref<16x256xf16, #pto.address_space<mat>>,
      %left: memref<16x256xf16, #pto.address_space<left>>,
      %right: memref<256x16xf16, #pto.address_space<right>>,
      %acc: memref<16x16xf32, #pto.address_space<acc>>,
      %ub_buf: memref<16x16xf32, #pto.address_space<vec>>,
      %gm_out: memref<16x16xf32, #pto.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      pto.tmov ins(%mat_a : memref<16x256xf16, #pto.address_space<mat>>)
               outs(%left : memref<16x256xf16, #pto.address_space<left>>)
      pto.tmatmul ins(%left, %right : memref<16x256xf16, #pto.address_space<left>>, memref<256x16xf16, #pto.address_space<right>>) outs(%acc : memref<16x16xf32, #pto.address_space<acc>>)
      pto.tstore ins(%ub_buf : memref<16x16xf32, #pto.address_space<vec>>) outs(%gm_out : memref<16x16xf32, #pto.address_space<gm>>)
    }
    return
  }
}

// CHECK: __DAV_CUBE__
// CHECK: for
// CHECK: TMOV
// CHECK: TMATMUL
// CHECK: __DAV_VEC__
// CHECK: for
// CHECK: TSTORE
