// RUN: ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=vpto --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > %t 2>&1
// RUN: FileCheck %s < %t
// RUN: ! rg 'IR Dump After (PTOValidateSimdIR|PTOInstantiateAndLowerToLibCall|PTOInlineLibCall|PTOVPTOConstIfCleanup)' %t

// CHECK-LABEL: IR Dump After FusionPlan
// CHECK-LABEL: IR Dump After OpScheduling
// CHECK-LABEL: IR Dump After PTOFusionRegionGen
// CHECK-LABEL: IR Dump After PTOLoweringSyncToPipe
// CHECK-LABEL: IR Dump After {anonymous}::InferPTOLayoutPass
// CHECK-LABEL: IR Dump After mlir::pto::{anonymous}::PTOViewToMemrefPass
// CHECK-LABEL: IR Dump After PlanMemory
// CHECK-LABEL: IR Dump After CSE
// CHECK-LABEL: IR Dump After PTOVPTOVersionSelection
// CHECK-LABEL: IR Dump After PTOToVPTO
// CHECK-LABEL: IR Dump After PTOVPTOIfCanonicalize
// CHECK-LABEL: IR Dump After PTOFusionMergeVecScope
// CHECK-LABEL: IR Dump After PTOLowLevelLoopFusion
// CHECK-LABEL: IR Dump After PTOFusionPredicateElision
// CHECK-LABEL: IR Dump After PTOFusionLoadStoreElision
// CHECK-LABEL: IR Dump After PTOPostFusionLoopUnroll
// CHECK-LABEL: IR Dump After PTOFlattenFusionRegion
// CHECK-LABEL: IR Dump After CSE
// CHECK-LABEL: IR Dump After PTOVPTOExpandBridgeOps
// CHECK-LABEL: IR Dump After CSE
