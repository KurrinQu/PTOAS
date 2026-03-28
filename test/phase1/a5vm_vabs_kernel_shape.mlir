// RUN: ./build/tools/ptoas/ptoas %s -o - | FileCheck %s

// CHECK-LABEL: @vabs_kernel
// CHECK: %[[MASK:.+]] = a5vm.pset_b32 "PAT_ALL" : !a5vm.mask
// CHECK: %[[LOAD:.+]] = a5vm.vlds %arg0[%arg2] : !pto.ptr<f32, ub> -> !a5vm.vec<64xf32>
// CHECK: %[[ABS:.+]] = a5vm.vabs %[[LOAD]], %[[MASK]] : !a5vm.vec<64xf32>, !a5vm.mask -> !a5vm.vec<64xf32>
// CHECK: a5vm.vsts %[[ABS]], %arg1[%arg2], %[[MASK]] : !a5vm.vec<64xf32>, !pto.ptr<f32, ub>, !a5vm.mask
module {
  func.func @vabs_kernel(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, ub>, %index: index) {
    %mask = a5vm.pset_b32 "PAT_ALL" : !a5vm.mask
    %tile = a5vm.vlds %src[%index] : !pto.ptr<f32, ub> -> !a5vm.vec<64xf32>
    %abs = a5vm.vabs %tile, %mask : !a5vm.vec<64xf32>, !a5vm.mask -> !a5vm.vec<64xf32>
    a5vm.vsts %abs, %dst[%index], %mask : !a5vm.vec<64xf32>, !pto.ptr<f32, ub>, !a5vm.mask
    return
  }
}

// CHECK: error: 'a5vm.vabs' op requires matching register vector shape
module {
  func.func @vabs_shape_mismatch(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, ub>, %index: index) {
    %mask = a5vm.pset_b32 "PAT_ALL" : !a5vm.mask
    %tile = a5vm.vlds %src[%index] : !pto.ptr<f32, ub> -> !a5vm.vec<64xf32>
    %abs = a5vm.vabs %tile, %mask : !a5vm.vec<64xf32>, !a5vm.mask -> !a5vm.vec<128xi16>
    a5vm.vsts %abs, %dst[%index], %mask : !a5vm.vec<128xi16>, !pto.ptr<f32, ub>, !a5vm.mask
    return
  }
}
