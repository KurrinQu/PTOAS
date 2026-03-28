// RUN: { ptoas %S/reduce_row_degenerate_layout_compat.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1 || true; } | FileCheck %s
// XFAIL: *

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-DAG: pto.oplib.instance.op = "trowsum"
// CHECK-DAG: pto.oplib.instance.variant_id = "linear"
// CHECK-DAG: pto.oplib.instance.op = "trowmin"
// CHECK-DAG: pto.oplib.instance.variant_id = "linear"
// CHECK-DAG: pto.oplib.instance.op = "trowmax"
// CHECK-DAG: pto.oplib.instance.variant_id = "linear"
