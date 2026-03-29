// RUN: ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=vpto --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After PTOLowLevelLoopFusion/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s --check-prefix=FUSED
// RUN: awk '/IR Dump After PTOVPTOExpandBridgeOps/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s --check-prefix=EXPAND

// FUSED-LABEL: IR Dump After PTOLowLevelLoopFusion
// FUSED: scf.for %{{.*}} = %c0{{(_[0-9]+)?}} to %c1{{(_[0-9]+)?}} step %c1{{(_[0-9]+)?}} {
// FUSED: scf.for %[[ROW:.+]] = %c0{{(_[0-9]+)?}} to %c16{{(_[0-9]+)?}} step %c1{{(_[0-9]+)?}} {
// FUSED: %[[UVLD0:.+]] = pto.uvld %{{.+}}#1[%{{.+}}] : memref<1x16xf32
// FUSED: %[[DUP0:.+]] = pto.vdup %[[UVLD0]]
// FUSED: %[[UVLD1:.+]] = pto.uvld %{{.+}}#2[%{{.+}}] : memref<1x16xf32
// FUSED: %[[DUP1:.+]] = pto.vdup %[[UVLD1]]
// FUSED: %[[CHUNK:.+]]:3 = scf.for %{{.*}} = %c0{{(_[0-9]+)?}} to %{{.+}} step %c1{{(_[0-9]+)?}} iter_args(%{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}) -> (i32, i32, i32) {
// FUSED: pto.vmul %{{.+}}, %[[DUP0]], %{{.+}}
// FUSED: pto.vmul %{{.+}}, %[[DUP1]], %{{.+}}
// FUSED: pto.vadd

// EXPAND-LABEL: IR Dump After PTOVPTOExpandBridgeOps
// EXPAND-NOT: pto.uvld
// EXPAND-NOT: llvm.getelementptr
// EXPAND: %[[BASE0:.+]] = pto.castptr %{{.+}} : memref<1x16xf32{{.*}} -> !pto.ptr<f32, ub>
// EXPAND: %[[PTR0:.+]] = pto.addptr %[[BASE0]], %{{.+}} : <f32, ub> -> <f32, ub>
// EXPAND: %[[ALIGN0:.+]] = pto.vldas %[[PTR0]] : !pto.ptr<f32, ub> -> !pto.align
// EXPAND: %[[LOAD0:.+]], %{{.+}}, %{{.+}} = pto.vldus %[[PTR0]], %[[ALIGN0]] : !pto.ptr<f32, ub>, !pto.align -> !pto.vec<64xf32>, !pto.align, !pto.ptr<f32, ub>
// EXPAND: pto.vdup %[[LOAD0]]
// EXPAND: %[[BASE1:.+]] = pto.castptr %{{.+}} : memref<1x16xf32{{.*}} -> !pto.ptr<f32, ub>
// EXPAND: %[[PTR1:.+]] = pto.addptr %[[BASE1]], %{{.+}} : <f32, ub> -> <f32, ub>
// EXPAND: %[[ALIGN1:.+]] = pto.vldas %[[PTR1]] : !pto.ptr<f32, ub> -> !pto.align
// EXPAND: %[[LOAD1:.+]], %{{.+}}, %{{.+}} = pto.vldus %[[PTR1]], %[[ALIGN1]] : !pto.ptr<f32, ub>, !pto.align -> !pto.vec<64xf32>, !pto.align, !pto.ptr<f32, ub>
// EXPAND: pto.vdup %[[LOAD1]]
