// RUN: ptoas --enable-cv-separation --pto-arch=a5 %s | FileCheck %s --check-prefix=A5
// RUN: ptoas --enable-cv-separation --pto-arch=a3 %s | FileCheck %s --check-prefix=A3

// Test: A5 generates set_intra_block/wait_intra_block, A3 generates ffts_cross_core_sync/wait_flag_dev.

func.func @cv_sync_test(
    %mat : memref<16x256xf16, #pto.address_space<mat>>,
    %left : memref<16x256xf16, #pto.address_space<left>>,
    %right : memref<256x16xf16, #pto.address_space<right>>,
    %acc : memref<16x16xf32, #pto.address_space<acc>>,
    %vec : memref<16x16xf32, #pto.address_space<vec>>,
    %gm_out : memref<16x16xf32, #pto.address_space<gm>>
) {
  pto.section.cube {
    pto.tmov ins(%mat : memref<16x256xf16, #pto.address_space<mat>>)
             outs(%left : memref<16x256xf16, #pto.address_space<left>>)
    pto.tmatmul ins(%left, %right : memref<16x256xf16, #pto.address_space<left>>,
                     memref<256x16xf16, #pto.address_space<right>>)
                outs(%acc : memref<16x16xf32, #pto.address_space<acc>>)
    pto.sync.set #pto.pipe<PIPE_FIX>, 0
  }
  pto.section.vector {
    pto.sync.wait #pto.pipe<PIPE_V>, 0
    pto.tstore ins(%vec : memref<16x16xf32, #pto.address_space<vec>>)
               outs(%gm_out : memref<16x16xf32, #pto.address_space<gm>>)
  }
  return
}

// A5: __DAV_CUBE__
// A5: TMATMUL
// A5: set_intra_block(PIPE_FIX, 0)
// A5: __DAV_VEC__
// A5: wait_intra_block(PIPE_V, 0)
// A5: TSTORE

// A3: __DAV_CUBE__
// A3: TMATMUL
// A3: ffts_cross_core_sync(PIPE_FIX, 0)
// A3: __DAV_VEC__
// A3: wait_flag_dev(PIPE_V, 0)
// A3: TSTORE
