// RUN: { ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null 2>&1 || true; } | awk '/IR Dump After PTOFusionLoadStoreElision/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOFusionLoadStoreElision/) exit; print}' | FileCheck %s

// Focused driver-sample regression for task 4.3:
// group 0 keeps stores for yielded frontier values that still leave the region
// through v1-conservative treshape boundaries, while internal-dead intermediates
// in the same region no longer materialize redundant maskedstores.

// CHECK-LABEL: IR Dump After PTOFusionLoadStoreElision
// CHECK-LABEL: func.func @kernel_online_update(
// CHECK: %[[YMAX:[0-9]+]] = pto.simd.tile_to_memref %15
// CHECK: %[[DEAD0:[0-9]+]] = pto.simd.tile_to_memref %17
// CHECK: %[[YEXP0:[0-9]+]] = pto.simd.tile_to_memref %19
// CHECK: %[[YEXP1:[0-9]+]] = pto.simd.tile_to_memref %21
// CHECK: %[[DEAD1:[0-9]+]] = pto.simd.tile_to_memref %23
// CHECK: %[[YSUM:[0-9]+]] = pto.simd.tile_to_memref %25
// CHECK: pto.simd.vec_scope {
// CHECK: vector.maskedstore %[[YMAX]]
// CHECK: vector.maskedstore %[[YEXP0]]
// CHECK: vector.maskedstore %[[YEXP1]]
// CHECK: vector.maskedstore %[[YSUM]]
// CHECK-NOT: vector.maskedstore %[[DEAD0]]
// CHECK-NOT: vector.maskedstore %[[DEAD1]]
// CHECK: pto.yield(%15, %19, %21, %25)
// CHECK: %[[GY0:[0-9]+]] = pto.simd.tile_to_memref %11
// CHECK: %[[GY1:[0-9]+]] = pto.simd.tile_to_memref %5
// CHECK: vector.maskedstore %[[GY0]]
// CHECK: vector.maskedstore %[[GY1]]
// CHECK: pto.yield(%11, %5)
