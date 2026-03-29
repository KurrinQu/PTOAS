// RUN: ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=vpto --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After PTOFusionPredicateElision/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s --check-prefix=POST

// POST-LABEL: IR Dump After PTOFusionPredicateElision
// POST: %[[LOOPRES:[^ ]+]] = scf.for %{{[^ ]+}} = %c0 to %c2 step %c1 iter_args(%[[ARG:[^ ]+]] = %c128_i32) -> (i32) {
// POST: %[[MASK:[^,]+]], %[[OUT:[^ ]+]] = pto.plt_b32 %[[ARG]] : i32 -> !pto.mask, i32
// POST-NOT: -> (i32, i32)
// POST: scf.yield %[[OUT]] : i32
