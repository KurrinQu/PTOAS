// RUN: ptoas %S/reduction_broadcast_family_emitc.pto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.cpp
// RUN: FileCheck %s --check-prefix=EMITC < %t.cpp
// XFAIL: *

// EMITC-LABEL: __global__ AICORE void reduction_broadcast_family_emitc_f32(
// EMITC-DAG: vcadd(
// EMITC-NOT: ptoas_vreduce_add(
