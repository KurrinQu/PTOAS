// RUN: { ptoas %S/softmax_chain.pto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK: pto.simd.tile_to_memref
// CHECK-LABEL: IR Dump After {anonymous}::EmitPTOManualPass
// CHECK: emitc.call_opaque "TASSIGN"
// CHECK: {{.*}}emitc.call_opaque "PTOAS__TILE_DATA"
