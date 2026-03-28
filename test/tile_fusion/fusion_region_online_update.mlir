// RUN: { ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-backend=a5vm --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null 2>&1 || true; } | FileCheck %s

// Driver-sample 5.5 region-encapsulation regression for paged_attention
// online_update: after PTOFusionRegionGen, the two main else-branch hotspots
// must each become one structured pto.fusion_region.

// CHECK-LABEL: IR Dump After PTOFusionRegionGen
// CHECK-LABEL: func.func @kernel_online_update(

// 1x16 diamond hotspot: one region carrying the softmax-like chain.
// CHECK: pto.fusion_region {
// CHECK: pto.tmax
// CHECK-NEXT: pto.tsub
// CHECK-NEXT: pto.texp
// CHECK-NEXT: pto.tsub
// CHECK-NEXT: pto.texp
// CHECK-NEXT: pto.tmul
// CHECK-NEXT: pto.tmul
// CHECK-NEXT: pto.tadd
// CHECK-NEXT: pto.yield(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!pto.tile_buf
// CHECK: {pto.fusion.group_id = 0 : i64}

// 16x128 join hotspot: one region carrying the two rowexpandmul producers and
// the final tadd join after the reshapes are hoisted.
// CHECK: %[[ROW0:[0-9]+]] = pto.treshape
// CHECK: %[[ROW1:[0-9]+]] = pto.treshape
// CHECK: pto.fusion_region {
// CHECK: pto.trowexpandmul ins(%{{[0-9]+}}, %[[ROW0]]
// CHECK-NEXT: pto.trowexpandmul ins(%{{[0-9]+}}, %[[ROW1]]
// CHECK-NEXT: pto.tadd ins(%{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: pto.yield(%{{.*}}, %{{.*}}) : (!pto.tile_buf
// CHECK: {pto.fusion.group_id = 1 : i64}
