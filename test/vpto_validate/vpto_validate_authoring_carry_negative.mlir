// RUN: ! ptoas --pto-backend=vpto --emit-vpto %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'pto.vaddc' op mask type '!pto.mask<b32>' does not match carry type '!pto.mask<b16>'
// CHECK: Error: VPTO authoring-stage legality verification failed.

module {
  func.func @carry_granularity_mismatch(%lhs: !pto.vreg<64xi32>,
                                        %rhs: !pto.vreg<64xi32>, %idx: index)
      attributes {pto.version_selection_applied} {
    pto.vecscope {
      %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
      %sum, %carry = pto.vaddc %lhs, %rhs, %mask
        : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32>
       -> !pto.vreg<64xi32>, !pto.mask<b16>
    }
    return
  }
}
