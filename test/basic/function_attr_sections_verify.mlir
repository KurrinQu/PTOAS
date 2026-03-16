// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @multi_block() attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %c0 = arith.constant 0 : i32
    cf.br ^bb1
  ^bb1:
    %c1 = arith.constant 1 : i32
    return
  }
}

// CHECK: error: 'func.func' op requires a single-block body for kernel_kind wrapping
