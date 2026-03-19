// RUN: { ptoas %S/../oplib/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOOutlineFusionGroups
// CHECK-LABEL: func.func private @__pto_fused_group_1_1(
// CHECK: pto.tadds
// CHECK: pto.tdivs
// CHECK: func.func private @__pto_fused_group_0_0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: memref<32x32xf32
// CHECK: pto.tmuls
// CHECK: pto.tmaxs
// CHECK: pto.tmins
// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-LABEL: func.func private @__pto_fused_group_1_1(
// CHECK: call @__pto_oplib_inst_l3_float_tile_scalar_template_tdivs_tile_scalar(
// CHECK-LABEL: func.func @flash_attention_softmax_block(
// CHECK: call @__pto_fused_group_0_0(
// CHECK: call @__pto_fused_group_1_1(
