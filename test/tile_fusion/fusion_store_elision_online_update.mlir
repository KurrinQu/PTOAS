// RUN: { ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null 2>&1 || true; } | awk '/IR Dump After PTOFusionLoadStoreElision/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOFusionLoadStoreElision/) exit; print}' | FileCheck %s

// A5VM-focused regression:
// - keep yielded frontier materialization (`%9` / `%17` stores remain)
// - eliminate round-trip reloads and dead intermediate stores inside the
//   straight-line fusion_region path when forwarding is provably mask-compatible.
// - normalize fusion-region yielded frontier to the underlying memrefs at
//   store-elision time, while re-binding region results for downstream users.

// CHECK-LABEL: IR Dump After PTOFusionLoadStoreElision
// CHECK-LABEL: func.func @kernel_online_update(
// CHECK: %[[REGION0:[0-9]+]]:4 = pto.fusion_region {
// CHECK: %[[STRAIGHT_SRC:[0-9]+]] = a5vm.vlds %4[%c0]
// CHECK: %[[STRAIGHT_SRC0:[0-9]+]] = a5vm.vlds %0[%c0]
// CHECK: %[[STRAIGHT_MAX:[0-9]+]] = a5vm.vmax %[[STRAIGHT_SRC]], %[[STRAIGHT_SRC0]], %mask
// CHECK: a5vm.vsts %[[STRAIGHT_MAX]], %9[%c0], %mask
// CHECK-NOT: a5vm.vlds %9[%c0]
// CHECK: %[[STRAIGHT_SUB:[0-9]+]] = a5vm.vsub %{{[0-9]+}}, %[[STRAIGHT_MAX]], %mask
// CHECK-NOT: a5vm.vlds %11[%c0]
// CHECK: %[[STRAIGHT_EXP0:[0-9]+]] = a5vm.vexp %[[STRAIGHT_SUB]], %mask {mode = "MODE_ZEROING"}
// CHECK: a5vm.vsts %[[STRAIGHT_EXP0]], %12[%c0], %mask
// CHECK-NOT: a5vm.vlds %16[%c0]
// CHECK-NOT: a5vm.vsts %{{[0-9]+}}, %11[%c0], %mask
// CHECK-NOT: a5vm.vsts %{{[0-9]+}}, %16[%c0], %mask
// CHECK: %[[STRAIGHT_ADD:[0-9]+]] = a5vm.vadd %{{[0-9]+}}, %{{[0-9]+}}, %mask
// CHECK: a5vm.vsts %[[STRAIGHT_ADD]], %17[%c0], %mask
// CHECK: pto.yield(%9, %12, %14, %17) : (memref<1x16xf32
// CHECK: } {pto.fusion.group_id = 0 : i64} : memref<1x16xf32
// CHECK: %[[REGION0_OUT3:[0-9]+]] = pto.bind_tile %[[REGION0]]#3, %c1, %c16
