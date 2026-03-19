// RUN: ptoas %S/compare_bitwise_family_emitc.pto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.cpp
// RUN: FileCheck %s --check-prefix=EMITC < %t.cpp
// XFAIL: *

// EMITC-LABEL: __global__ AICORE void compare_bitwise_family_emitc(
// EMITC-DAG: __VEC_SCOPE__ {
// EMITC-DAG: ptoas_vcmp(
// EMITC-DAG: vsel(
// EMITC-NOT: MaskReg {{[[:alnum:]_]+}} = {{[[:alnum:]_]+}} != {{[[:alnum:]_]+}};
