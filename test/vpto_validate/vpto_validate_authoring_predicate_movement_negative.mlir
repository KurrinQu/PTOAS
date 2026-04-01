// RUN: ! ptoas --pto-backend=vpto --emit-vpto %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'pto.pdintlv_b8' op operand #0 must be PTO low-level b8 mask type, but got '!pto.mask<b16>'

module {
  func.func @predicate_movement_mismatch(%idx: index)
      attributes {pto.version_selection_applied} {
    pto.vecscope {
      %lhs = pto.pset_b16 "PAT_ALL" : !pto.mask<b16>
      %rhs = pto.pge_b16 "PAT_ALL" : !pto.mask<b16>
      %low, %high = pto.pdintlv_b8 %lhs, %rhs
        : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<b16>
    }
    return
  }
}
