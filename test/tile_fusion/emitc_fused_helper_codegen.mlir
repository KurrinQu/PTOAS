// RUN: ptoas %S/../oplib/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.cpp
// RUN: FileCheck %s < %t.cpp

// CHECK-LABEL: [aicore] inline __attribute__((always_inline)) void __pto_fused_group_1_1(
// CHECK: __VEC_SCOPE__ {
// CHECK: vlds(
// CHECK: vdiv(
// CHECK: vsts(
// CHECK-NOT: vsel(
// CHECK-LABEL: [aicore] inline __attribute__((always_inline)) void __pto_fused_group_0_0(
// CHECK-LABEL: AICORE void flash_attention_softmax_block(
// CHECK: Tile<{{.*}}> [[TILE:v[0-9]+]];
// CHECK: TASSIGN([[TILE]], [[ADDR:v[0-9]+]]);
// CHECK: __ubuf__ float* [[PTR:v[0-9]+]] = [[TILE]].data();
