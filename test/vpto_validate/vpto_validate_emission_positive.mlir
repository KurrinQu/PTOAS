// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto %s -o - 2>/dev/null | FileCheck %s --check-prefix=IR
// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-text %s -o /dev/null
// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-llvm %s -o /dev/null

// IR-LABEL: func.func @memref_boundary_kernel(
// IR-SAME: %arg0: !pto.ptr<f32, ub>
// IR-SAME: %arg1: !pto.ptr<f32, ub>
// IR-SAME: %arg2: index
// IR: %[[MASK:.+]] = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
// IR: %[[LOAD:.+]] = pto.vlds %arg0[%arg2] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
// IR: pto.vsts %[[LOAD]], %arg1[%arg2], %[[MASK]] : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
// IR-NOT: pto.castptr %arg0
// IR-NOT: pto.castptr %arg1

module {
  func.func @memref_boundary_kernel(
      %src: memref<256xf32, #pto.address_space<vec>>,
      %dst: memref<256xf32, #pto.address_space<vec>>,
      %offset: index) attributes {pto.version_selection_applied} {
    pto.vecscope {
      %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
      %v = pto.vlds %src[%offset] : memref<256xf32, #pto.address_space<vec>> -> !pto.vreg<64xf32>
      pto.vsts %v, %dst[%offset], %mask : !pto.vreg<64xf32>, memref<256xf32, #pto.address_space<vec>>, !pto.mask<b32>
    }
    return
  }
}
