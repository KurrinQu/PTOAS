// RUN: ! ptoas --pto-backend=vpto --emit-vpto %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'pto.vcmp' op seed mask type '!pto.mask<b16>' does not match input vector type '!pto.vreg<64xf32>'; expected !pto.mask<b32>
// CHECK: Error: VPTO authoring-stage legality verification failed.

module {
  func.func @compare_granularity_mismatch(%src: !pto.ptr<f32, ub>,
                                          %dst: !pto.ptr<f32, ub>, %idx: index)
      attributes {pto.version_selection_applied} {
    pto.vecscope {
      %seed = pto.pset_b16 "PAT_ALL" : !pto.mask<b16>
      %lhs = pto.vlds %src[%idx] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
      %rhs = pto.vlds %dst[%idx] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
      %cmp = pto.vcmp %lhs, %rhs, %seed, "lt"
        : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b16> -> !pto.mask<b32>
      pto.vsts %lhs, %dst[%idx], %cmp : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
    }
    return
  }
}
