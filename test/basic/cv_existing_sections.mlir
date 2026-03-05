// RUN: ptoas --enable-cv-separation %s | FileCheck %s

// Test: existing section ops are preserved as-is.

module {
  func.func @existing_sections(
      %mat_a: memref<16x256xf16, #pto.address_space<mat>>,
      %left: memref<16x256xf16, #pto.address_space<left>>,
      %right: memref<256x16xf16, #pto.address_space<right>>,
      %acc: memref<16x16xf32, #pto.address_space<acc>>,
      %ub_buf: memref<16x16xf32, #pto.address_space<vec>>,
      %gm_out: memref<16x16xf32, #pto.address_space<gm>>) {
    pto.section.cube {
      pto.tmov ins(%mat_a : memref<16x256xf16, #pto.address_space<mat>>)
               outs(%left : memref<16x256xf16, #pto.address_space<left>>)
      pto.tmatmul ins(%left, %right : memref<16x256xf16, #pto.address_space<left>>, memref<256x16xf16, #pto.address_space<right>>) outs(%acc : memref<16x16xf32, #pto.address_space<acc>>)
    }
    pto.tstore ins(%ub_buf : memref<16x16xf32, #pto.address_space<vec>>) outs(%gm_out : memref<16x16xf32, #pto.address_space<gm>>)
    return
  }
}

// CHECK: pto.section.cube
// CHECK:   pto.tmov
// CHECK:   pto.tmatmul
// CHECK: pto.section.vector
// CHECK:   pto.tstore
