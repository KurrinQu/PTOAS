// RUN: ptoas --pto-cv-classify-and-split %s | FileCheck %s

// Test: pure vector ops go into section.vector, pure cube ops into section.cube.

module {
  func.func @pure_vector(%gm_in: memref<16x256xf16, #pto.address_space<gm>>,
                          %ub_buf: memref<16x256xf16, #pto.address_space<vec>>,
                          %gm_out: memref<16x256xf16, #pto.address_space<gm>>) {
    pto.tload ins(%gm_in : memref<16x256xf16, #pto.address_space<gm>>)
              outs(%ub_buf : memref<16x256xf16, #pto.address_space<vec>>)
    pto.tstore ins(%ub_buf : memref<16x256xf16, #pto.address_space<vec>>)
               outs(%gm_out : memref<16x256xf16, #pto.address_space<gm>>)
    return
  }
}

// CHECK: pto.section.vector
// CHECK-NOT: pto.section.cube
