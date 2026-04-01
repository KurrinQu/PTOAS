// RUN: ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=vpto --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After PTOToVPTO/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s --check-prefix=LOWER
// RUN: awk '/IR Dump After PTOPostFusionLoopUnroll/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s --check-prefix=UNROLL

// LOWER-LABEL: IR Dump After PTOToVPTO
// LOWER: pto.vecscope {
// LOWER: scf.for
// LOWER-NOT: llvm.loop.aivector_scope

// UNROLL-LABEL: IR Dump After PTOPostFusionLoopUnroll
// UNROLL: pto.vecscope {
// UNROLL: scf.for
// UNROLL-NOT: llvm.loop.aivector_scope
