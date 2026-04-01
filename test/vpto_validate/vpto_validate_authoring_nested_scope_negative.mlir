// RUN: ! ptoas --pto-backend=vpto --emit-vpto %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'pto.vecscope' op does not allow nested dedicated pto.vecscope/pto.strict_vecscope
// CHECK: Error: VPTO authoring-stage legality verification failed.

module {
  func.func @nested_vec_scope(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, ub>,
                              %idx: index) attributes {pto.version_selection_applied} {
    pto.vecscope {
      pto.vecscope {
        %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
        %v = pto.vlds %src[%idx] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
        pto.vsts %v, %dst[%idx], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
      }
    }
    return
  }
}
