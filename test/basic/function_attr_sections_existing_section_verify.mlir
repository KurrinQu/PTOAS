// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @already_sectioned() attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    pto.section.cube {
      %c1 = arith.constant 1 : i32
    }
    return
  }
}

// CHECK: error: 'func.func' op already contains pto.section.cube or pto.section.vector
