// RUN: { ptoas %S/../oplib/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all --print-ir-after-all-func-filter=__pto_fused_group_1_1 -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOLowLevelLoopFusion
// CHECK-LABEL: func.func private @__pto_fused_group_1_1(
// CHECK: %[[SRC0:[0-9]+]] = pto.simd.tile_to_memref %arg0
// CHECK: %[[SRC1:[0-9]+]] = pto.simd.tile_to_memref %arg1
// CHECK: %[[DST:[0-9]+]] = pto.simd.tile_to_memref %arg6
// CHECK: %[[SRC2:[0-9]+]] = pto.simd.tile_to_memref %arg2
// CHECK: %[[SRC3:[0-9]+]] = pto.simd.tile_to_memref %arg3
// CHECK-COUNT-1: pto.simd.vec_scope
// CHECK-COUNT-2: scf.for
// CHECK: vector.maskedload
// CHECK: arith.addf
// CHECK: arith.divf
// CHECK-NOT: vector.maskedload %[[DST]]
// CHECK-COUNT-1: vector.maskedstore
