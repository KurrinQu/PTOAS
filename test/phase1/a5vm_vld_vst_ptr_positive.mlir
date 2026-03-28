// RUN: ptoas --pto-backend=a5vm %s -o - | FileCheck %s

// CHECK-LABEL: @vld_vst_ptr_positive
// CHECK: %[[MASK_ALL:.+]] = a5vm.pset_b32 "PAT_ALL" : !a5vm.mask
// CHECK: %[[ALIGN:.+]] = a5vm.vldas %{{.+}} : !llvm.ptr<6> -> !a5vm.align
// CHECK: %[[V0:.+]] = a5vm.vlds %{{.+}} : !llvm.ptr<6> -> !a5vm.vec<64xf32>
// CHECK: %[[V1:.+]] = a5vm.vldus %[[ALIGN]], %{{.+}} : !a5vm.align, !llvm.ptr<6> -> !a5vm.vec<64xf32>
// CHECK: %[[PMASK:.+]] = a5vm.plds %{{.+}} : !llvm.ptr<6> -> !a5vm.mask
// CHECK: a5vm.vsts %[[V0]], %{{.+}}, %[[MASK_ALL]] : !a5vm.vec<64xf32>, !llvm.ptr<6>, !a5vm.mask
// CHECK: a5vm.vsts %[[V1]], %{{.+}}, %[[PMASK]] : !a5vm.vec<64xf32>, !llvm.ptr<6>, !a5vm.mask
// CHECK: a5vm.psts %[[MASK_ALL]], %{{.+}} : !a5vm.mask, !llvm.ptr<6>
module {
  func.func @vld_vst_ptr_positive(%src: !llvm.ptr<6>, %dst: !llvm.ptr<6>,
                                  %index: index) {
    %mask_all = a5vm.pset_b32 "PAT_ALL" : !a5vm.mask
    %align = a5vm.vldas %src[%index] : !llvm.ptr<6> -> !a5vm.align
    %v0 = a5vm.vlds %src[%index] : !llvm.ptr<6> -> !a5vm.vec<64xf32>
    %v1 = a5vm.vldus %align, %src[%index] : !a5vm.align, !llvm.ptr<6> -> !a5vm.vec<64xf32>
    %pmask = a5vm.plds %src[%index] : !llvm.ptr<6> -> !a5vm.mask
    a5vm.vsts %v0, %dst[%index], %mask_all : !a5vm.vec<64xf32>, !llvm.ptr<6>, !a5vm.mask
    a5vm.vsts %v1, %dst[%index], %pmask : !a5vm.vec<64xf32>, !llvm.ptr<6>, !a5vm.mask
    a5vm.psts %mask_all, %dst[%index] : !a5vm.mask, !llvm.ptr<6>
    return
  }
}
