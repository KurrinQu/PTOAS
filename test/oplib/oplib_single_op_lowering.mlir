// RUN: { ptoas %S/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s --check-prefix=IR
// RUN: { ptoas %S/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s --check-prefix=PRUNE

// IR-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// IR-LABEL: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tmul_{{.*}}(
// IR-LABEL: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tadd_{{.*}}(
// IR-LABEL: func.func private @__pto_fused_group_0_0(
// IR: call @__pto_oplib_inst_l3_float_binary_elementwise_template_tmul_{{.*}}(
// IR-LABEL: func.func @flash_attention_softmax_block(
// IR: call @__pto_fused_group_0_0(

// PRUNE-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// PRUNE-NOT: @__pto_oplib_seed_
// PRUNE-NOT: @__pto_oplib_variant_
// PRUNE-NOT: pto.oplib.entry_role
