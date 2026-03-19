// RUN: { ptoas %S/../oplib/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all --print-ir-after-all-func-filter=flash_attention_softmax_block -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOCreateFusionGroups
// CHECK-LABEL: func.func @flash_attention_softmax_block(
// CHECK: pto.tmuls{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 0 : i64
// CHECK: pto.tmaxs{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 1 : i64
// CHECK: pto.tmins{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 2 : i64
// CHECK: pto.tmul{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 3 : i64
// CHECK: pto.tmul{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 4 : i64
// CHECK: pto.tmul{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 5 : i64
// CHECK: pto.tadd{{.*}}pto.fusion.group_id = 1 : i64{{.*}}pto.fusion.order = 0 : i64
// CHECK: pto.tadd{{.*}}pto.fusion.group_id = 1 : i64{{.*}}pto.fusion.order = 1 : i64
// CHECK: pto.tadd{{.*}}pto.fusion.group_id = 1 : i64{{.*}}pto.fusion.order = 2 : i64
// CHECK: pto.tadds{{.*}}pto.fusion.group_id = 1 : i64{{.*}}pto.fusion.order = 3 : i64
// CHECK: pto.tdivs{{.*}}pto.fusion.group_id = 1 : i64{{.*}}pto.fusion.order = 4 : i64
