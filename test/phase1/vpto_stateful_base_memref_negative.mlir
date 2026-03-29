// RUN: bash -lc 'set +e; ptoas --pto-backend=vpto %s -o - 2>&1; echo EXIT:$?' | FileCheck %s

// CHECK: error: 'pto.pstu' op operand #2 must be LLVM pointer type
// CHECK: EXIT:1
module {
  func.func @pstu_memref_base_should_fail(
      %src: memref<256xf32, #pto.address_space<vec>>,
      %base: memref<256xf32, #pto.address_space<vec>>,
      %index: index) {
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
    %align = pto.vldas %src[%index] : memref<256xf32, #pto.address_space<vec>> -> !pto.align
    %next_align, %next_base = pto.pstu %align, %mask, %base : !pto.align, !pto.mask, memref<256xf32, #pto.address_space<vec>> -> !pto.align, !llvm.ptr<6>
    return
  }
}
