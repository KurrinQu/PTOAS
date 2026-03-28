// RUN: { ptoas %S/generic_shape_64x32_f32_dynamic_vshape.pto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1 || true; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tadd_{{.*}}(
// CHECK: memref<64x32xf32{{.*}}#pto.address_space<vec>>
// CHECK: pto.simd.tile_to_memref
// CHECK-LABEL: IR Dump After {anonymous}::EmitPTOManualPass
// CHECK: emitc.call_opaque "TASSIGN"
// CHECK: {{.*}}emitc.call_opaque "PTOAS__TILE_DATA"
