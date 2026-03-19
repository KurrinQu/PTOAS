// RUN: { ptoas %S/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all --print-ir-after-all-func-filter=__pto_fused_group_0_0 -o /dev/null 2>&1; } | sed -n '/IR Dump After PTOInstantiateAndLowerToLibCall/,/IR Dump After PTOInlineLibCall/p' | FileCheck %s --check-prefix=MIXED
// RUN: { ptoas %S/binary_max_min_chain.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all --print-ir-after-all-func-filter=__pto_fused_group_0_0 -o /dev/null 2>&1; } | sed -n '/IR Dump After PTOInstantiateAndLowerToLibCall/,/IR Dump After PTOInlineLibCall/p' | FileCheck %s --check-prefix=PURE

// MIXED-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// MIXED-LABEL: func.func private @__pto_fused_group_0_0(
// MIXED-SAME: %arg0: f32, %arg1: f32, %arg2: f32
// MIXED: @__pto_oplib_inst_l3_float_tile_scalar_template_tmuls_tile_scalar
// MIXED: @__pto_oplib_inst_l3_float_tile_scalar_template_tmaxs_tile_scalar
// MIXED: @__pto_oplib_inst_l3_float_tile_scalar_template_tmins_tile_scalar
// MIXED-COUNT-3: @__pto_oplib_inst_l3_float_binary_elementwise_template_tmul_tile

// PURE-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// PURE-LABEL: func.func private @__pto_fused_group_0_0(
// PURE-NOT: float_tile_scalar_template
// PURE: @__pto_oplib_inst_l3_float_binary_elementwise_template_tmul_tile
// PURE: @__pto_oplib_inst_l3_float_binary_elementwise_template_tdiv_tile
// PURE: @__pto_oplib_inst_l3_float_binary_elementwise_template_tadd_tile
// PURE: @__pto_oplib_inst_l3_float_binary_elementwise_template_tsub_tile
// PURE: @__pto_oplib_inst_l3_float_binary_elementwise_template_tmax_tile
// PURE: @__pto_oplib_inst_l3_float_binary_elementwise_template_tmin_tile
