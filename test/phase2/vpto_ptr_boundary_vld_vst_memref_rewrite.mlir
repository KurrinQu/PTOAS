// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-text --print-ir-after-all %s -o /dev/null 2>&1 | sed -n '/IR Dump After PTOVPTOPtrBoundary/,/IR Dump After /p' | FileCheck %s

// CHECK-LABEL: IR Dump After PTOVPTOPtrBoundary
// CHECK: func.func @memref_vld_vst_boundary(%arg0: !pto.ptr<f32, ub>, %arg1: !pto.ptr<f32, ub>, %arg2: index, %arg3: !pto.mask)
// CHECK-NOT: pto.castptr %arg0
// CHECK-NOT: pto.castptr %arg1
// CHECK: %[[LOAD:.+]] = pto.vlds %arg0[%arg2] : !pto.ptr<f32, ub> -> !pto.vec<64xf32>
// CHECK: pto.vsts %[[LOAD]], %arg1[%arg2], %arg3 : !pto.vec<64xf32>, !pto.ptr<f32, ub>, !pto.mask

module {
  func.func @memref_vld_vst_boundary(
      %src: memref<256xf32, #pto.address_space<vec>>,
      %dst: memref<256xf32, #pto.address_space<vec>>,
      %offset: index, %mask: !pto.mask) {
    %v = pto.vlds %src[%offset] : memref<256xf32, #pto.address_space<vec>> -> !pto.vec<64xf32>
    pto.vsts %v, %dst[%offset], %mask : !pto.vec<64xf32>, memref<256xf32, #pto.address_space<vec>>, !pto.mask
    return
  }
}
