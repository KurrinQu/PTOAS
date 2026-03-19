// RUN: rm -rf %t.bad && mkdir -p %t.bad
// RUN: cp %S/resources/bad_memref_template.txt %t.bad/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.bad --pto-arch=a5 -o %t.bad.cpp > %t.bad.log 2>&1
// RUN: FileCheck %s < %t.bad.log

// CHECK: invalid OP-Lib signature for kind=l3_binary_elementwise_template
