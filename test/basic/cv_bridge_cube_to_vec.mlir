// RUN: ptoas --enable-cv-separation %s | FileCheck %s

// Test: cube and vector ops are separated into different sections.
// No cross-domain SSA deps because all ops are DPS format.

module {
  func.func @bridge_test(
      %left: memref<16x256xf16, #pto.address_space<left>>,
      %right: memref<256x16xf16, #pto.address_space<right>>,
      %acc: memref<16x16xf32, #pto.address_space<acc>>,
      %ub_buf: memref<16x16xf32, #pto.address_space<vec>>,
      %gm_out: memref<16x16xf32, #pto.address_space<gm>>,
      %workspace: memref<16x16xf32, #pto.address_space<gm>>) {
    pto.tmatmul ins(%left, %right : memref<16x256xf16, #pto.address_space<left>>, memref<256x16xf16, #pto.address_space<right>>) outs(%acc : memref<16x16xf32, #pto.address_space<acc>>)
    pto.tstore ins(%ub_buf : memref<16x16xf32, #pto.address_space<vec>>) outs(%gm_out : memref<16x16xf32, #pto.address_space<gm>>)
    return
  }
}

// CHECK: __DAV_CUBE__
// CHECK: TMATMUL
// CHECK: __DAV_VEC__
// CHECK: TSTORE
