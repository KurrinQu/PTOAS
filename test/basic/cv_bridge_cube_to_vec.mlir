// RUN: ptoas --pto-cv-classify-and-split --pto-cv-insert-bridge %s | FileCheck %s

// Test: cross-domain value gets bridged through workspace.
// Cube section should have tstore + sync.set, vector section should have sync.wait + tload.

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

// CHECK: pto.section.cube
// CHECK:   pto.tmatmul
// CHECK: pto.section.vector
// CHECK:   pto.tstore
