// RUN: ptoas --enable-cv-separation %s | FileCheck %s

// Test: pure vector ops go into __DAV_VEC__ section only.

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

// CHECK: __DAV_VEC__
// CHECK: TLOAD
// CHECK: TSTORE
// CHECK-NOT: __DAV_CUBE__
