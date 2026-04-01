// RUN: ! ptoas --pto-backend=vpto --emit-vpto %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'pto.pset_b32' op result #0 must be PTO low-level b32 mask type, but got '!pto.mask<b16>'

module {
  func.func @family_suffix_mismatch(%idx: index)
      attributes {pto.version_selection_applied} {
    pto.vecscope {
      %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b16>
    }
    return
  }
}
