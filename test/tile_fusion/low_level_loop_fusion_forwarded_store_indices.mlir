// RUN: { ptoas %S/../oplib/binary_max_min_chain.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all --print-ir-after-all-func-filter=__pto_fused_group_0_0 -o /dev/null 2>&1; } | sed -n '/IR Dump After PTOLowLevelLoopFusion/,/IR Dump After Canonicalizer/p' | FileCheck %s

// CHECK-LABEL: IR Dump After PTOLowLevelLoopFusion
// CHECK-LABEL: func.func private @__pto_fused_group_0_0(
// CHECK-COUNT-1: pto.simd.vec_scope
// CHECK-COUNT-2: scf.for
// CHECK: %[[MAX:.*]] = arith.maximumf
// CHECK-NOT: vector.maskedload %arg5
// CHECK: arith.minimumf %[[MAX]]
