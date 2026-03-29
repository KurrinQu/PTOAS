// RUN: ptoas --pto-backend=vpto %s -o - | FileCheck %s

// CHECK-LABEL: @vld_vst_ptr_positive
// CHECK: %[[MASK_ALL:.+]] = pto.pset_b32 "PAT_ALL" : !pto.mask
// CHECK: %[[ALIGN:.+]] = pto.vldas %{{.+}} : !llvm.ptr<6> -> !pto.align
// CHECK: %[[V0:.+]] = pto.vlds %{{.+}} : !llvm.ptr<6> -> !pto.vec<64xf32>
// CHECK: %[[V1:.+]] = pto.vldus %[[ALIGN]], %{{.+}} : !pto.align, !llvm.ptr<6> -> !pto.vec<64xf32>
// CHECK: %[[PMASK:.+]] = pto.plds %{{.+}} : !llvm.ptr<6> -> !pto.mask
// CHECK: pto.vsts %[[V0]], %{{.+}}, %[[MASK_ALL]] : !pto.vec<64xf32>, !llvm.ptr<6>, !pto.mask
// CHECK: pto.vsts %[[V1]], %{{.+}}, %[[PMASK]] : !pto.vec<64xf32>, !llvm.ptr<6>, !pto.mask
// CHECK: pto.psts %[[MASK_ALL]], %{{.+}} : !pto.mask, !llvm.ptr<6>
module {
  func.func @vld_vst_ptr_positive(%src: !llvm.ptr<6>, %dst: !llvm.ptr<6>,
                                  %index: index) {
    %mask_all = pto.pset_b32 "PAT_ALL" : !pto.mask
    %align = pto.vldas %src[%index] : !llvm.ptr<6> -> !pto.align
    %v0 = pto.vlds %src[%index] : !llvm.ptr<6> -> !pto.vec<64xf32>
    %v1 = pto.vldus %align, %src[%index] : !pto.align, !llvm.ptr<6> -> !pto.vec<64xf32>
    %pmask = pto.plds %src[%index] : !llvm.ptr<6> -> !pto.mask
    pto.vsts %v0, %dst[%index], %mask_all : !pto.vec<64xf32>, !llvm.ptr<6>, !pto.mask
    pto.vsts %v1, %dst[%index], %pmask : !pto.vec<64xf32>, !llvm.ptr<6>, !pto.mask
    pto.psts %mask_all, %dst[%index] : !pto.mask, !llvm.ptr<6>
    return
  }
}
