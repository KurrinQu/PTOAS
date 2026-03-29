// RUN: bash -lc 'set +e; ptoas --pto-backend=vpto %s -o - 2>&1; echo EXIT:$?' | FileCheck %s

// CHECK: error: 'pto.pstu' op result #1 must be LLVM pointer type
// CHECK: EXIT:1
module {
  func.func @pstu_memref_base_out_should_fail(
      %src: !llvm.ptr<6>, %base: !llvm.ptr<6>, %index: index) {
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
    %align = pto.vldas %src[%index] : !llvm.ptr<6> -> !pto.align
    %next_align, %next_base = pto.pstu %align, %mask, %base : !pto.align, !pto.mask, !llvm.ptr<6> -> !pto.align, memref<256xf32, #pto.address_space<vec>>
    return
  }
}
