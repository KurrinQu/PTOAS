// RUN: bash -lc 'set +e; ptoas --pto-backend=a5vm %s -o - 2>&1; echo EXIT:$?' | FileCheck %s

// CHECK: error: 'a5vm.pstu' op result #1 must be LLVM pointer type
// CHECK: EXIT:1
module {
  func.func @pstu_memref_base_out_should_fail(
      %src: !llvm.ptr<6>, %base: !llvm.ptr<6>, %index: index) {
    %mask = a5vm.pset_b32 "PAT_ALL" : !a5vm.mask
    %align = a5vm.vldas %src[%index] : !llvm.ptr<6> -> !a5vm.align
    %next_align, %next_base = a5vm.pstu %align, %mask, %base : !a5vm.align, !a5vm.mask, !llvm.ptr<6> -> !a5vm.align, memref<256xf32, #pto.address_space<vec>>
    return
  }
}
