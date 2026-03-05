// RUN: ptoas --enable-cv-separation %s | FileCheck %s

// Test: no cross-domain dependency — no sync ops inserted.

module {
  func.func @no_cross_dep(
      %gm_a: memref<16x256xf16, #pto.address_space<gm>>,
      %mat_a: memref<16x256xf16, #pto.address_space<mat>>,
      %left: memref<16x256xf16, #pto.address_space<left>>,
      %right: memref<256x16xf16, #pto.address_space<right>>,
      %acc: memref<16x16xf32, #pto.address_space<acc>>,
      %ub_buf: memref<16x16xf32, #pto.address_space<vec>>,
      %gm_out: memref<16x16xf32, #pto.address_space<gm>>,
      %workspace: memref<16x16xf32, #pto.address_space<gm>>) {
    pto.tload ins(%gm_a : memref<16x256xf16, #pto.address_space<gm>>)
              outs(%mat_a : memref<16x256xf16, #pto.address_space<mat>>)
    pto.tmov ins(%mat_a : memref<16x256xf16, #pto.address_space<mat>>)
             outs(%left : memref<16x256xf16, #pto.address_space<left>>)
    pto.tmatmul ins(%left, %right : memref<16x256xf16, #pto.address_space<left>>, memref<256x16xf16, #pto.address_space<right>>) outs(%acc : memref<16x16xf32, #pto.address_space<acc>>)
    pto.tstore ins(%ub_buf : memref<16x16xf32, #pto.address_space<vec>>) outs(%gm_out : memref<16x16xf32, #pto.address_space<gm>>)
    return
  }
}

// CHECK: __DAV_CUBE__
// CHECK: __DAV_VEC__
// CHECK-NOT: set_flag
// CHECK-NOT: wait_flag
