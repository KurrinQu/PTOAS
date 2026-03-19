// RUN: ptoas %S/../oplib/binary_max_min_chain.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.cpp
// RUN: FileCheck %s < %t.cpp

// CHECK-LABEL: [aicore] inline __attribute__((always_inline)) void __pto_fused_group_0_0(
// CHECK: vmax(
// CHECK: vmin(
