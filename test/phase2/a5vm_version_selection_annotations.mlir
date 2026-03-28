// RUN: ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After PTOA5VMVersionSelection/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s

// CHECK-LABEL: IR Dump After PTOA5VMVersionSelection
// CHECK: func.func @kernel_online_update{{.*}}attributes {pto.a5vm_version_selection_applied}
// CHECK-DAG: pto.tload{{.*}}pto.a5vm_lowering_choice = #pto.a5vm_lowering_choice<update_mode = post_update, loop_shape = two_d>
// CHECK-DAG: pto.tstore{{.*}}pto.a5vm_lowering_choice = #pto.a5vm_lowering_choice<update_mode = post_update, loop_shape = two_d>
// CHECK-DAG: pto.trowexpanddiv{{.*}}pto.a5vm_lowering_choice = #pto.a5vm_lowering_choice<update_mode = post_update, loop_shape = two_d>
// CHECK-DAG: pto.texpands{{.*}}pto.a5vm_lowering_choice = #pto.a5vm_lowering_choice<update_mode = post_update, loop_shape = two_d>
// CHECK-DAG: pto.tmax{{.*}}pto.a5vm_lowering_choice = #pto.a5vm_lowering_choice<update_mode = no_post_update, loop_shape = two_d>
// CHECK-DAG: pto.tmul{{.*}}pto.a5vm_lowering_choice = #pto.a5vm_lowering_choice<update_mode = no_post_update, loop_shape = two_d>
// CHECK-DAG: pto.trowexpandmul{{.*}}pto.a5vm_lowering_choice = #pto.a5vm_lowering_choice<update_mode = no_post_update, loop_shape = two_d>
// CHECK-DAG: pto.tadd{{.*}}pto.a5vm_lowering_choice = #pto.a5vm_lowering_choice<update_mode = no_post_update, loop_shape = two_d>
