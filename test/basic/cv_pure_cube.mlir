// RUN: ptoas --enable-cv-separation %s | FileCheck %s

// Test: complete cube data path — tload GM→MAT, tmov MAT→LEFT/RIGHT, tmatmul, all in section.cube.

module {
  func.func @cube_full_path(
      %gm_a: memref<16x256xf16, #pto.address_space<gm>>,
      %gm_b: memref<256x16xf16, #pto.address_space<gm>>,
      %mat_a: memref<16x256xf16, #pto.address_space<mat>>,
      %mat_b: memref<256x16xf16, #pto.address_space<mat>>,
      %left: memref<16x256xf16, #pto.address_space<left>>,
      %right: memref<256x16xf16, #pto.address_space<right>>,
      %acc: memref<16x16xf32, #pto.address_space<acc>>) {
    pto.tload ins(%gm_a : memref<16x256xf16, #pto.address_space<gm>>)
              outs(%mat_a : memref<16x256xf16, #pto.address_space<mat>>)
    pto.tload ins(%gm_b : memref<256x16xf16, #pto.address_space<gm>>)
              outs(%mat_b : memref<256x16xf16, #pto.address_space<mat>>)
    pto.tmov ins(%mat_a : memref<16x256xf16, #pto.address_space<mat>>)
             outs(%left : memref<16x256xf16, #pto.address_space<left>>)
    pto.tmov ins(%mat_b : memref<256x16xf16, #pto.address_space<mat>>)
             outs(%right : memref<256x16xf16, #pto.address_space<right>>)
    pto.tmatmul ins(%left, %right : memref<16x256xf16, #pto.address_space<left>>, memref<256x16xf16, #pto.address_space<right>>) outs(%acc : memref<16x16xf32, #pto.address_space<acc>>)
    return
  }
}

// CHECK: pto.section.cube
// CHECK:   pto.tload
// CHECK:   pto.tload
// CHECK:   pto.tmov
// CHECK:   pto.tmov
// CHECK:   pto.tmatmul
// CHECK-NOT: pto.section.vector
