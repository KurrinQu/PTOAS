// RUN: { ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-backend=a5vm --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null 2>&1 || true; } | FileCheck %s

// Validate the two main online_update planning hotspots:
// 1. the 1x16 diamond softmax-like subgraph
// 2. the 16x128 join subgraph formed by two trowexpandmul producers and a tadd

// CHECK-LABEL: IR Dump After FusionPlan
// CHECK-LABEL: func.func @kernel_online_update(

// 1x16 diamond hotspot
// CHECK: pto.tmax{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 0 : i64
// CHECK: pto.tsub{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 1 : i64
// CHECK: pto.texp{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 2 : i64
// CHECK: pto.tsub{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 3 : i64
// CHECK: pto.texp{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 4 : i64
// CHECK: pto.tmul{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 5 : i64
// CHECK: pto.tmul{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 6 : i64
// CHECK: pto.tadd{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 7 : i64

// 16x128 join hotspot
// CHECK: pto.trowexpandmul{{.*}}pto.fusion.group_id = 1 : i64{{.*}}pto.fusion.order = 0 : i64
// CHECK: pto.trowexpandmul{{.*}}pto.fusion.group_id = 1 : i64{{.*}}pto.fusion.order = 1 : i64
// CHECK: pto.tadd{{.*}}pto.fusion.group_id = 1 : i64{{.*}}pto.fusion.order = 2 : i64
