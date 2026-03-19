// RUN: { ptoas %S/generic_shape_64x32_f32.pto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-LABEL: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tmul_{{.*}}(
// CHECK-LABEL: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tadd_{{.*}}(
// CHECK-LABEL: func.func @binary_chain_f32_64x32(
// CHECK: call @__pto_oplib_inst_l3_float_binary_elementwise_template_tadd_{{.*}}(
// CHECK: call @__pto_oplib_inst_l3_float_binary_elementwise_template_tmul_{{.*}}(
// CHECK: memref<64x32xf32{{.*}}#pto.address_space<vec>>
