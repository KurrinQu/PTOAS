// RUN: ptoas --pto-arch=a5 --pto-backend=a5vm --a5vm-emit-hivm-text --print-ir-after-all %s -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: IR Dump After PTOA5VMPtrBoundary
// CHECK: func.func @memref_boundary_kernel(%arg0: !pto.ptr<f32, ub>, %arg1: !pto.ptr<f32, ub>, %arg2: index, %arg3: !a5vm.mask)
// CHECK-NOT: pto.castptr %arg0
// CHECK-NOT: pto.castptr %arg1
// CHECK: %[[LOAD:.+]] = a5vm.vlds %arg0[%arg2] : !pto.ptr<f32, ub> -> !a5vm.vec<64xf32>
// CHECK: a5vm.vsts %[[LOAD]], %arg1[%arg2], %arg3 : !a5vm.vec<64xf32>, !pto.ptr<f32, ub>, !a5vm.mask

module {
  func.func @memref_boundary_kernel(
      %src: memref<256xf32, #pto.address_space<vec>>,
      %dst: memref<256xf32, #pto.address_space<vec>>,
      %offset: index, %mask: !a5vm.mask) {
    %v = a5vm.vlds %src[%offset] : memref<256xf32, #pto.address_space<vec>> -> !a5vm.vec<64xf32>
    a5vm.vsts %v, %dst[%offset], %mask : !a5vm.vec<64xf32>, memref<256xf32, #pto.address_space<vec>>, !a5vm.mask
    return
  }
}
