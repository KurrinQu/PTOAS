// RUN: { ptoas %S/reduction_broadcast_family_static.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1 || true; } | FileCheck %s
// XFAIL: *

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_reduce_row_template_trowsum_linear(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_reduce_row_template_trowmax_linear(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_reduce_row_template_trowmin_linear(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_reduce_col_template_tcolmax_linear(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_reduce_col_template_tcolmin_linear(
