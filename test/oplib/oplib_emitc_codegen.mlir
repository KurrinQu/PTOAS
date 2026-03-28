// RUN: ptoas %S/binary_max_min_chain.pto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.bin6.cpp
// RUN: FileCheck %s --check-prefix=BIN6 < %t.bin6.cpp
// RUN: ptoas %S/softmax_chain.pto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.softmax.cpp
// RUN: FileCheck %s --check-prefix=SOFTMAX < %t.softmax.cpp
// RUN: rm -rf %t.attr && mkdir -p %t.attr
// RUN: cp %S/../../oplib/level3/float_binary_elementwise_templates.mlir %t.attr/float_binary_elementwise_templates.mlir
// RUN: sed -i 's/pto.simd.vld_dist = \"NORM\"/pto.simd.vld_dist = \"NORM_USER_TOKEN\"/g' %t.attr/float_binary_elementwise_templates.mlir
// RUN: sed -i 's/pto.simd.vst_dist = \"DIST_NORM\"/pto.simd.vst_dist = \"DIST_USER_TOKEN\"/g' %t.attr/float_binary_elementwise_templates.mlir
// RUN: sed -i 's/pto.simd.exec_mode = \"MODE_ZEROING\"/pto.simd.exec_mode = \"MODE_USER_TOKEN\"/g' %t.attr/float_binary_elementwise_templates.mlir
// RUN: ptoas %S/binary_max_min_chain.pto --op-lib-dir=%t.attr --pto-arch=a5 -o %t.attr.cpp
// RUN: FileCheck %s --check-prefix=ATTR < %t.attr.cpp

// BIN6-DAG: vlds(
// BIN6-DAG: vsts(
// BIN6-DAG: vadd(
// BIN6-DAG: vsub(
// BIN6-DAG: vmul(
// BIN6-DAG: vdiv(
// BIN6-DAG: vmax(
// BIN6-DAG: vmin(
// BIN6-DAG: CreatePredicate<float>(
// BIN6-DAG: __VEC_SCOPE__ {
// BIN6-NOT: __VEC_SCOPE__ {;
// BIN6-DAG: for (uint16_t
// BIN6-DAG: for (uint16_t {{.*}} = 0; {{.*}} < 32; {{.*}} += 1) {
// BIN6-NOT: for (uint16_t {{.*}} = 0; {{.*}} < {{.*}}; {{.*}} += 64)
// BIN6-NOT: vlds_mask(
// BIN6-NOT: PTOAS__OPLIB_

// SOFTMAX: vlds(
// SOFTMAX: vsts(
// SOFTMAX: vdup(
// SOFTMAX: vadd(
// SOFTMAX: CreatePredicate<float>(
// SOFTMAX-DAG: __VEC_SCOPE__ {
// SOFTMAX-NOT: __VEC_SCOPE__ {;
// SOFTMAX-NOT: vlds_mask(
// SOFTMAX-NOT: PTOAS__OPLIB_

// ATTR: NORM_USER_TOKEN
// ATTR: DistVST::DIST_USER_TOKEN
// ATTR: MODE_USER_TOKEN
