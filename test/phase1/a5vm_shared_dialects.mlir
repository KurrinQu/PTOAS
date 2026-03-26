// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm --emit-a5vm %s -o - 2>/dev/null | FileCheck %s

// This fixture pins the expectation that scalar math and loop structure stay
// in shared dialects even when the hardware-facing operation is an a5vm op.
// CHECK: arith.addi
// CHECK: scf.for
// CHECK: scf.yield
// CHECK: a5vm.vabs
module {
  func.func @shared_dialects(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, ub>, %arg1: index, %arg2: index) -> index {
    %sum = arith.addi %arg1, %arg2 : index
    %loop = scf.for %iv = %arg1 to %arg2 step %arg1 iter_args(%acc = %sum) -> (index) {
      %next = arith.addi %acc, %iv : index
      scf.yield %next : index
    }
    %mask = a5vm.pset_b32 "PAT_ALL" : !a5vm.mask
    %0 = a5vm.vlds %src[%arg1] : !pto.ptr<f32, ub> -> !a5vm.vec<64xf32>
    %1 = a5vm.vabs %0, %mask : !a5vm.vec<64xf32>, !a5vm.mask -> !a5vm.vec<64xf32>
    a5vm.vsts %1, %dst[%arg1], %mask : !a5vm.vec<64xf32>, !pto.ptr<f32, ub>, !a5vm.mask
    return %loop : index
  }
}

// CHECK-NOT: llvm.hivm
