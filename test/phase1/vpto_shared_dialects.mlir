// RUN: ./build/tools/ptoas/ptoas --pto-backend=vpto --emit-vpto %s -o - 2>/dev/null | FileCheck %s

// This fixture pins the expectation that scalar math and loop structure stay
// in shared dialects even when the hardware-facing operation is an vpto op.
// CHECK: arith.addi
// CHECK: scf.for
// CHECK: scf.yield
// CHECK: pto.vabs
module {
  func.func @shared_dialects(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, ub>, %arg1: index, %arg2: index) -> index {
    %sum = arith.addi %arg1, %arg2 : index
    %loop = scf.for %iv = %arg1 to %arg2 step %arg1 iter_args(%acc = %sum) -> (index) {
      %next = arith.addi %acc, %iv : index
      scf.yield %next : index
    }
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
    %0 = pto.vlds %src[%arg1] : !pto.ptr<f32, ub> -> !pto.vec<64xf32>
    %1 = pto.vabs %0, %mask : !pto.vec<64xf32>, !pto.mask -> !pto.vec<64xf32>
    pto.vsts %1, %dst[%arg1], %mask : !pto.vec<64xf32>, !pto.ptr<f32, ub>, !pto.mask
    return %loop : index
  }
}

// CHECK-NOT: llvm.hivm
