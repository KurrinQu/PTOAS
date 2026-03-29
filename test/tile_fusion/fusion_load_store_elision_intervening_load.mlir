// RUN: { ptoas %s --enable-op-fusion --pto-backend=vpto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all --print-ir-after-all-func-filter=__pto_fused_group_13_13 -o /dev/null 2>&1; } | sed -n '/IR Dump After PTOFusionLoadStoreElision/,/IR Dump After Canonicalizer/p' | FileCheck %s

// CHECK-LABEL: IR Dump After PTOFusionLoadStoreElision
// CHECK-LABEL: func.func private @__pto_fused_group_13_13(
// CHECK-COUNT-1: pto.simd.vec_scope
// CHECK-COUNT-1: scf.for
// CHECK: vector.maskedload %arg0
// CHECK: vector.maskedstore %arg1
// CHECK: vector.maskedload %arg2
// CHECK: vector.maskedload %arg1
// CHECK: arith.subf
// CHECK: vector.maskedstore %arg3

module {
  func.func private @__pto_fused_group_13_13(%arg0: memref<32x96xf32>, %arg1: memref<32x96xf32>, %arg2: memref<32x96xf32>, %arg3: memref<32x96xf32>) attributes {pto.fusion.group_id = 13 : i64} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c96 = arith.constant 96 : index
    pto.simd.vec_scope {
      %zero = arith.constant dense<0.000000e+00> : vector<64xf32>
      scf.for %i = %c0 to %c32 step %c1 {
        scf.for %j = %c0 to %c96 step %c64 {
          %remain = arith.subi %c96, %j : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %lanes = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %lanes : vector<64xi1>
          %lhs = vector.maskedload %arg0[%i, %j], %mask, %zero : memref<32x96xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %max = arith.maximumf %lhs, %lhs : vector<64xf32>
          vector.maskedstore %arg1[%i, %j], %mask, %max : memref<32x96xf32>, vector<64xi1>, vector<64xf32>
          %other = vector.maskedload %arg2[%i, %j], %mask, %zero : memref<32x96xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %roundtrip = vector.maskedload %arg1[%i, %j], %mask, %zero : memref<32x96xf32>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %diff = arith.subf %other, %roundtrip : vector<64xf32>
          vector.maskedstore %arg3[%i, %j], %mask, %diff : memref<32x96xf32>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    return
  }
}
