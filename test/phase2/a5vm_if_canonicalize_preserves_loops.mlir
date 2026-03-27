// RUN: ptoas --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --print-ir-after-all --print-ir-after-all-func-filter=cleanup_preserves_loops %s -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After PTOA5VMIfCanonicalize/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s

// CHECK-LABEL: IR Dump After PTOA5VMIfCanonicalize
// CHECK: func.func @cleanup_preserves_loops(%arg0: i1) -> i32
// CHECK: %[[OUTER:.+]] = scf.for %{{.*}} = %c0 to %c8 step %c1 iter_args
// CHECK: %[[SEL:.+]] = arith.select %arg0, %c1_i32, %c2_i32 : i32
// CHECK: %[[INNER:.+]] = scf.for %{{.*}} = %c0 to %c2 step %c1 iter_args
// CHECK-NOT: scf.if

module {
  func.func @cleanup_preserves_loops(%cond: i1) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true

    %outer = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %c0_i32) -> (i32) {
      %dynamic = scf.if %cond -> (i32) {
        scf.yield %c1_i32 : i32
      } else {
        scf.yield %c2_i32 : i32
      }
      %folded = scf.if %true -> (i32) {
        scf.yield %dynamic : i32
      } else {
        scf.yield %c2_i32 : i32
      }
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%innerAcc = %folded) -> (i32) {
        %nextInner = arith.addi %innerAcc, %dynamic : i32
        scf.yield %nextInner : i32
      }
      %next = arith.addi %acc, %inner : i32
      scf.yield %next : i32
    }

    return %outer : i32
  }
}
