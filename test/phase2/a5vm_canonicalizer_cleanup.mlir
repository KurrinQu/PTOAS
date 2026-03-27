// RUN: ! ptoas --pto-arch=a5 --pto-backend=emitc --print-ir-after-all --print-ir-after-all-func-filter=cleanup_smoke %s -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After Canonicalizer/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s

// CHECK-LABEL: IR Dump After Canonicalizer
// CHECK: func.func @cleanup_smoke(%arg0: i1) -> i32 {
// CHECK-NOT: scf.if
// CHECK-NOT: %true
// CHECK: %[[SEL:[^ ]+]] = arith.select %arg0, %c1_i32, %c2_i32 : i32
// CHECK: %[[SUM:[^ ]+]] = arith.addi %[[SEL]], %c1_i32 : i32
// CHECK: return %[[SUM]] : i32

module {
  func.func @cleanup_smoke(%cond: i1) -> i32 {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %true = arith.constant true
    %folded = scf.if %true -> (i32) {
      scf.yield %c1 : i32
    } else {
      scf.yield %c2 : i32
    }
    %dynamic = scf.if %cond -> (i32) {
      scf.yield %c1 : i32
    } else {
      scf.yield %c2 : i32
    }
    %sum = arith.addi %folded, %dynamic : i32
    return %sum : i32
  }
}
