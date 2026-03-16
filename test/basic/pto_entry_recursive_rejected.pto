// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @kernel(%arg0: i32) -> i32 attributes {pto.entry} {
    %0 = func.call @kernel(%arg0) : (i32) -> i32
    return %0 : i32
  }
}

// CHECK: recursive function calls are not supported for EmitC C++ emission
