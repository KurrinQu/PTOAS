// RUN: { ptoas %S/generic_shape_dtype_chain.pto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tadd_{{.*}}(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tmul_{{.*}}(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tadd_tile_f16_scalar(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tmin_tile_f16_scalar(
// CHECK-NOT: func.func private @__pto_oplib_entry_
// CHECK-NOT: no matching OP-Lib entry
