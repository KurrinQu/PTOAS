// RUN: ! ptoas --pto-backend=vpto --emit-vpto %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'scf.for' op legacy VPTO authoring form is no longer accepted; use dedicated pto.vecscope/pto.strict_vecscope instead of scf.for with 'llvm.loop.aivector_scope'
// CHECK: Error: VPTO authoring-stage legality verification failed.

module {
  func.func @legacy_vec_scope_rejected(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, ub>,
                                       %idx: index) attributes {pto.version_selection_applied} {
    %c1 = arith.constant 1 : index
    scf.for %i = %idx to %c1 step %c1 {
      %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
      %v = pto.vlds %src[%idx] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
      pto.vsts %v, %dst[%idx], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
    } {llvm.loop.aivector_scope}
    return
  }
}
