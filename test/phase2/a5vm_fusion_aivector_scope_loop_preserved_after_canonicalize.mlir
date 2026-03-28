// RUN: ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After Canonicalizer/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s

// CHECK-LABEL: IR Dump After Canonicalizer
// CHECK: func.func @kernel_online_update
// CHECK: scf.for %{{.*}} = %c0{{(_[0-9]+)?}} to %c1{{(_[0-9]+)?}} step %c1{{(_[0-9]+)?}} {
// CHECK: a5vm.plt_b32
// CHECK: a5vm.vdup
// CHECK: a5vm.vsts
// CHECK: } {llvm.loop.aivector_scope}

module {
}
