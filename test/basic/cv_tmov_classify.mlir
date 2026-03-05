// RUN: ptoas --enable-cv-separation %s | FileCheck %s

// Test: TMOV classified by address space — MAT→LEFT is CUBE, VEC→VEC is VECTOR.

module {
  func.func @tmov_classify(
      %mat_buf: memref<16x256xf16, #pto.address_space<mat>>,
      %left: memref<16x256xf16, #pto.address_space<left>>,
      %ub_src: memref<16x256xf16, #pto.address_space<vec>>,
      %ub_dst: memref<16x256xf16, #pto.address_space<vec>>) {
    // This tmov touches MAT and LEFT → should be CUBE
    pto.tmov ins(%mat_buf : memref<16x256xf16, #pto.address_space<mat>>)
             outs(%left : memref<16x256xf16, #pto.address_space<left>>)
    // This tmov touches only VEC → should be VECTOR
    pto.tmov ins(%ub_src : memref<16x256xf16, #pto.address_space<vec>>)
             outs(%ub_dst : memref<16x256xf16, #pto.address_space<vec>>)
    return
  }
}

// CHECK: pto.section.cube
// CHECK:   pto.tmov
// CHECK: pto.section.vector
// CHECK:   pto.tmov
