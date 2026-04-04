// RUN: ! ptoas --pto-backend=vpto --emit-vpto %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'pto.vlds' op requires exactly one linearized index when source is !pto.ptr
// CHECK: Error: Failed to parse MLIR.

module {
  func.func @ptr_multidim_is_illegal(%src: !pto.ptr<f32, ub>, %row: index,
                                     %col: index)
      attributes {pto.version_selection_applied} {
    pto.vecscope {
      %v = pto.vlds %src[%row, %col] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    }
    return
  }
}
