// RUN: ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=vpto --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After PTOToVPTO/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s

// CHECK-LABEL: IR Dump After PTOToVPTO
// CHECK: func.func @kernel_online_update
// CHECK: %[[REPEAT_UPPER:.+]] = arith.ceildivui %c128{{(_[0-9]+)?}}, %c64{{(_[0-9]+)?}} : index
// CHECK: %[[ROW_SCALAR:.+]] = arith.index_castui %c128{{(_[0-9]+)?}} : index to i32
// CHECK: scf.for %{{.*}} = %c0{{(_[0-9]+)?}} to %c1{{(_[0-9]+)?}} step %c1{{(_[0-9]+)?}} {
// CHECK: scf.for %{{.*}} = %c0{{(_[0-9]+)?}} to %c16{{(_[0-9]+)?}} step %c1{{(_[0-9]+)?}} {
// CHECK: %[[BROADCAST:.+]] = pto.uvld
// CHECK: pto.vdup
// CHECK: scf.for %[[CHUNK:.+]] = %c0{{(_[0-9]+)?}} to %[[REPEAT_UPPER]] step %c1{{(_[0-9]+)?}} iter_args(%[[REMAIN:.+]] = %[[ROW_SCALAR]]) -> (i32) {
// CHECK: %[[MASK:.+]], %[[NEXT:.+]] = pto.plt_b32 %[[REMAIN]]
// CHECK: %{{.+}} = arith.muli %[[CHUNK]], %c64{{(_[0-9]+)?}} : index
// CHECK: pto.vlds
// CHECK: pto.vmul {{.*}}, %[[MASK]]
// CHECK: pto.vsts {{.*}}, %[[MASK]]
// CHECK: scf.yield %[[NEXT]] : i32
