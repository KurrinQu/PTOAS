// RUN: { ptoas %s --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all --print-ir-after-all-func-filter=__pto_fused_group_12_12 -o /dev/null 2>&1; } | sed -n '/IR Dump After PTOLowLevelLoopFusion/,/IR Dump After Canonicalizer/p' | FileCheck %s

// CHECK-LABEL: IR Dump After PTOLowLevelLoopFusion
// CHECK-LABEL: func.func private @__pto_fused_group_12_12(
// CHECK-COUNT-1: pto.simd.vec_scope
// CHECK: scf.for %{{.*}} = %c0 to %c1 step %c1
// CHECK: scf.for %{{.*}} = %c0 to %c32 step %c64
// CHECK-NOT: vector.maskedload %arg1
// CHECK-COUNT-1: vector.maskedstore %arg2

module {
  func.func private @__pto_fused_group_12_12(%arg0: memref<1x32xf32>, %arg1: memref<1x32xf32>, %arg2: memref<1x32xf32>) attributes {pto.fusion.group_id = 12 : i64} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    pto.simd.vec_scope {
      %zero = arith.constant dense<0.000000e+00> : vector<64xf32>
      scf.for %i = %c0 to %c1 step %c1 {
        scf.for %j = %c0 to %c32 step %c64 {
          %mask = vector.constant_mask [32] : vector<64xi1>
          %v = vector.maskedload %arg0[%i, %j], %mask, %zero : memref<1x32xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %sum = arith.addf %v, %v : vector<64xf32>
          vector.maskedstore %arg1[%i, %j], %mask, %sum : memref<1x32xf32>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    pto.simd.vec_scope {
      %zero = arith.constant dense<0.000000e+00> : vector<64xf32>
      scf.for %i = %c0 to %c1 step %c1 {
        scf.for %j = %c0 to %c32 step %c64 {
          %mask = vector.constant_mask [32] : vector<64xi1>
          %v = vector.maskedload %arg1[%i, %j], %mask, %zero : memref<1x32xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %sum = arith.addf %v, %v : vector<64xf32>
          vector.maskedstore %arg2[%i, %j], %mask, %sum : memref<1x32xf32>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    return
  }
}
