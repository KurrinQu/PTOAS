// RUN: ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > %t 2>&1 || true
// RUN: awk '/IR Dump After PTOToA5VM/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s
// RUN: ! rg 'A5VM ptr-only boundary failed' %t

// CHECK-LABEL: IR Dump After PTOToA5VM
// CHECK: func.func @kernel_online_update(%arg0: memref<?xf32
// CHECK: %[[TMP0:.+]] = pto.pointer_cast
// CHECK-SAME: : memref<16x1xf32
// CHECK: a5vm.copy_gm_to_ubuf
