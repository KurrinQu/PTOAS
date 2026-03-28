// RUN: ptoas %S/generic_shape_dtype_chain.pto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.generic.cpp
// RUN: FileCheck %s < %t.generic.cpp

// CHECK-DAG: AICORE void binary_chain_f16_64x64(
// CHECK-DAG: AICORE void binary_chain_f32_48x48(
// CHECK-DAG: vadd(
// CHECK-DAG: vmin(
// CHECK-DAG: MaskReg {{v[0-9]+}} = CreatePredicate<half>({{v[0-9]+}});
// CHECK: uint32_t {{v[0-9]+}} = 0;
// CHECK: {{v[0-9]+}} = {{v[0-9]+}};
// CHECK-DAG: MaskReg {{v[0-9]+}} = CreatePredicate<float>({{v[0-9]+}});
// CHECK-NOT: TADD(
// CHECK-NOT: TMIN(
// CHECK-NOT: PTOAS__OPLIB_
