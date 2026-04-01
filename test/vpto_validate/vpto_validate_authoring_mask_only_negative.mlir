// RUN: ! ptoas --pto-backend=vpto --emit-vpto %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'pto.pnot' op input mask type '!pto.mask<b32>' does not match mask type '!pto.mask<b16>'
// CHECK: Error: VPTO authoring-stage legality verification failed.

module {
  func.func @mask_only_mismatch(%idx: index)
      attributes {pto.version_selection_applied} {
    pto.vecscope {
      %in = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
      %gate = pto.pset_b16 "PAT_ALL" : !pto.mask<b16>
      %out = pto.pnot %in, %gate : !pto.mask<b32>, !pto.mask<b16> -> !pto.mask<b32>
    }
    return
  }
}
