// RUN: { ptoas %S/binary_max_min_chain.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOLowLevelLoopFusion
// CHECK-LABEL: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tmin_{{.*}}(
// CHECK-SAME: pto.oplib.instance.from_seed = false
// CHECK-SAME: pto.oplib.instance.op = "tmin"
// CHECK-LABEL: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tmax_{{.*}}(
// CHECK-SAME: pto.oplib.instance.from_seed = false
// CHECK-SAME: pto.oplib.instance.op = "tmax"
// CHECK-LABEL: func.func private @__pto_fused_group_0_0(
// CHECK: arith.maximumf
// CHECK: arith.minimumf
