// RUN: ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=vpto --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After PTOVPTOVersionSelection/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s

// CHECK-LABEL: IR Dump After PTOVPTOVersionSelection
// CHECK: func.func @kernel_online_update{{.*}}attributes {pto.version_selection_applied}
// CHECK-DAG: pto.tload{{.*}}pto.lowering_choice = #pto.lowering_choice<update_mode = post_update, loop_shape = two_d>
// CHECK-DAG: pto.tstore{{.*}}pto.lowering_choice = #pto.lowering_choice<update_mode = post_update, loop_shape = two_d>
// CHECK-DAG: pto.trowexpanddiv{{.*}}pto.lowering_choice = #pto.lowering_choice<update_mode = post_update, loop_shape = two_d>
// CHECK-DAG: pto.texpands{{.*}}pto.lowering_choice = #pto.lowering_choice<update_mode = post_update, loop_shape = two_d>
// CHECK-DAG: pto.tmax{{.*}}pto.lowering_choice = #pto.lowering_choice<update_mode = no_post_update, loop_shape = two_d>
// CHECK-DAG: pto.tmul{{.*}}pto.lowering_choice = #pto.lowering_choice<update_mode = no_post_update, loop_shape = two_d>
// CHECK-DAG: pto.trowexpandmul{{.*}}pto.lowering_choice = #pto.lowering_choice<update_mode = no_post_update, loop_shape = two_d>
// CHECK-DAG: pto.tadd{{.*}}pto.lowering_choice = #pto.lowering_choice<update_mode = no_post_update, loop_shape = two_d>
