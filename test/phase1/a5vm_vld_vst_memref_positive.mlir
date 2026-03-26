// RUN: ptoas --pto-backend=a5vm %s -o - | FileCheck %s

// CHECK-LABEL: @vld_vst_memref_positive
// CHECK: %[[MASK_ALL:.+]] = a5vm.pset_b32 "PAT_ALL" : !a5vm.mask
// CHECK: %[[ALIGN:.+]] = a5vm.vldas %{{.+}} : memref<256xf32> -> !a5vm.align
// CHECK: %[[V0:.+]] = a5vm.vlds %{{.+}} : memref<256xf32> -> !a5vm.vec<64xf32>
// CHECK: %[[V1:.+]] = a5vm.vldus %[[ALIGN]], %{{.+}} : !a5vm.align, memref<256xf32> -> !a5vm.vec<64xf32>
// CHECK: %[[PMASK:.+]] = a5vm.plds %{{.+}} : memref<256xf32> -> !a5vm.mask
// CHECK: a5vm.vsts %[[V0]], %{{.+}}, %[[MASK_ALL]] : !a5vm.vec<64xf32>, memref<256xf32>, !a5vm.mask
// CHECK: a5vm.vsts %[[V1]], %{{.+}}, %[[PMASK]] : !a5vm.vec<64xf32>, memref<256xf32>, !a5vm.mask
// CHECK: a5vm.psts %[[MASK_ALL]], %{{.+}} : !a5vm.mask, memref<256xf32>
module {
  func.func @vld_vst_memref_positive(%src: memref<256xf32>, %dst: memref<256xf32>,
                                     %index: index) {
    %mask_all = a5vm.pset_b32 "PAT_ALL" : !a5vm.mask
    %align = a5vm.vldas %src[%index] : memref<256xf32> -> !a5vm.align
    %v0 = a5vm.vlds %src[%index] : memref<256xf32> -> !a5vm.vec<64xf32>
    %v1 = a5vm.vldus %align, %src[%index] : !a5vm.align, memref<256xf32> -> !a5vm.vec<64xf32>
    %pmask = a5vm.plds %src[%index] : memref<256xf32> -> !a5vm.mask
    a5vm.vsts %v0, %dst[%index], %mask_all : !a5vm.vec<64xf32>, memref<256xf32>, !a5vm.mask
    a5vm.vsts %v1, %dst[%index], %pmask : !a5vm.vec<64xf32>, memref<256xf32>, !a5vm.mask
    a5vm.psts %mask_all, %dst[%index] : !a5vm.mask, memref<256xf32>
    return
  }
}
