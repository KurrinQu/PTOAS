// RUN: ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=vpto --post-fusion-loop-unroll-factor=2 --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After PTOFusionLoadStoreElision/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s --check-prefix=LS
// RUN: awk '/IR Dump After PTOPostFusionLoopUnroll/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s --check-prefix=UNROLL

// Driver-sample guard for the new unroll forwarding option:
// on paged_attention online_update, forcing x2 must rewrite the short
// softmax-like fusion-region carrier loop after load/store cleanup.

// LS-LABEL: func.func @kernel_online_update(
// LS: %49:4 = pto.fusion_region {
// LS: pto.vecscope {
// LS: scf.for %arg9 = %c0 to %c1 step %c1 {
// LS: pto.vmax
// LS: pto.vexp
// LS: pto.vadd
// LS-NOT: %c2_1 = arith.constant 2 : index

// UNROLL-LABEL: func.func @kernel_online_update(
// UNROLL: %49:4 = pto.fusion_region {
// UNROLL: pto.vecscope {
// UNROLL: %c0_0 = arith.constant 0 : index
// UNROLL: %c2_1 = arith.constant 2 : index
// UNROLL: scf.for %arg9 = %c0 to %c0_0 step %c2_1 {
// UNROLL: pto.vmax
// UNROLL: pto.vexp
// UNROLL: pto.vadd
// UNROLL: %c1_4 = arith.constant 1 : index
// UNROLL: pto.vmax
// UNROLL: pto.vexp
// UNROLL: pto.vadd
