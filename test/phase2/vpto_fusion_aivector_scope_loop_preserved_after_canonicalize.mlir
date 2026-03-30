// RUN: ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=vpto --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After Canonicalizer/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s

// CHECK-LABEL: IR Dump After Canonicalizer
// CHECK: func.func @kernel_online_update
// CHECK: pto.vecscope {
// CHECK: pto.plt_b32
// CHECK: pto.vdup
// CHECK: pto.vsts
// CHECK: }

module {
}
