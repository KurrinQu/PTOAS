// RUN: ! ptoas --pto-backend=vpto --emit-vpto %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'pto.vabs' op mask type '!pto.mask<b16>' does not match input vector type '!pto.vreg<64xf32>'; expected !pto.mask<b32>
// CHECK: Error: VPTO authoring-stage legality verification failed.

module {
  func.func @mask_granularity_mismatch(%src: !pto.ptr<f32, ub>,
                                       %dst: !pto.ptr<f32, ub>, %idx: index)
      attributes {pto.version_selection_applied} {
    pto.vecscope {
      %mask = pto.pset_b16 "PAT_ALL" : !pto.mask<b16>
      %v = pto.vlds %src[%idx] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
      %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b16> -> !pto.vreg<64xf32>
      pto.vsts %abs, %dst[%idx], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b16>
    }
    return
  }
}
