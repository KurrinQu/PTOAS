// RUN: ptoas --enable-cv-separation --pto-arch=a5 %s | FileCheck %s

// Test: A5 pipeline works with odd-row dimensions.
// The odd-row error would trigger only if a cross-section SSA dep exists.
// With DPS ops, no bridge is needed, so pipeline succeeds.

func.func @odd_rows(
    %left : memref<15x256xf16, #pto.address_space<left>>,
    %right : memref<256x15xf16, #pto.address_space<right>>,
    %acc : memref<15x15xf32, #pto.address_space<acc>>,
    %vec : memref<15x15xf32, #pto.address_space<vec>>,
    %gm_out : memref<15x15xf32, #pto.address_space<gm>>
) {
  pto.section.cube {
    pto.tmatmul ins(%left, %right : memref<15x256xf16, #pto.address_space<left>>,
                     memref<256x15xf16, #pto.address_space<right>>)
                outs(%acc : memref<15x15xf32, #pto.address_space<acc>>)
  }
  pto.section.vector {
    pto.tstore ins(%vec : memref<15x15xf32, #pto.address_space<vec>>)
               outs(%gm_out : memref<15x15xf32, #pto.address_space<gm>>)
  }
  return
}

// CHECK: __DAV_CUBE__
// CHECK: TMATMUL
// CHECK: __DAV_VEC__
// CHECK: TSTORE
