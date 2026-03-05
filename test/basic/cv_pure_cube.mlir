// RUN: ptoas --enable-cv-separation %s | FileCheck %s

// Test: pure cube ops go into section.cube only.

module {
  func.func @pure_cube(
      %left: memref<16x256xf16, #pto.address_space<left>>,
      %right: memref<256x16xf16, #pto.address_space<right>>,
      %acc: memref<16x16xf32, #pto.address_space<acc>>) {
    pto.tmatmul ins(%left, %right : memref<16x256xf16, #pto.address_space<left>>, memref<256x16xf16, #pto.address_space<right>>) outs(%acc : memref<16x16xf32, #pto.address_space<acc>>)
    return
  }
}

// CHECK: pto.section.cube
// CHECK-NOT: pto.section.vector
