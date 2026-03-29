// RUN: ptoas --pto-backend=vpto %s -o - | FileCheck %s

// CHECK-LABEL: @vld_vst_memref_positive
// CHECK: %[[MASK_ALL:.+]] = pto.pset_b32 "PAT_ALL" : !pto.mask
// CHECK: %[[ALIGN:.+]] = pto.vldas %{{.+}} : memref<256xf32> -> !pto.align
// CHECK: %[[V0:.+]] = pto.vlds %{{.+}} : memref<256xf32> -> !pto.vec<64xf32>
// CHECK: %[[V1:.+]] = pto.vldus %[[ALIGN]], %{{.+}} : !pto.align, memref<256xf32> -> !pto.vec<64xf32>
// CHECK: %[[PMASK:.+]] = pto.plds %{{.+}} : memref<256xf32> -> !pto.mask
// CHECK: pto.vsts %[[V0]], %{{.+}}, %[[MASK_ALL]] : !pto.vec<64xf32>, memref<256xf32>, !pto.mask
// CHECK: pto.vsts %[[V1]], %{{.+}}, %[[PMASK]] : !pto.vec<64xf32>, memref<256xf32>, !pto.mask
// CHECK: pto.psts %[[MASK_ALL]], %{{.+}} : !pto.mask, memref<256xf32>
module {
  func.func @vld_vst_memref_positive(%src: memref<256xf32>, %dst: memref<256xf32>,
                                     %index: index) {
    %mask_all = pto.pset_b32 "PAT_ALL" : !pto.mask
    %align = pto.vldas %src[%index] : memref<256xf32> -> !pto.align
    %v0 = pto.vlds %src[%index] : memref<256xf32> -> !pto.vec<64xf32>
    %v1 = pto.vldus %align, %src[%index] : !pto.align, memref<256xf32> -> !pto.vec<64xf32>
    %pmask = pto.plds %src[%index] : memref<256xf32> -> !pto.mask
    pto.vsts %v0, %dst[%index], %mask_all : !pto.vec<64xf32>, memref<256xf32>, !pto.mask
    pto.vsts %v1, %dst[%index], %pmask : !pto.vec<64xf32>, memref<256xf32>, !pto.mask
    pto.psts %mask_all, %dst[%index] : !pto.mask, memref<256xf32>
    return
  }
}
