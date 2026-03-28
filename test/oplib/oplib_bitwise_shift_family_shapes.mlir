// RUN: rm -rf %t.dir && mkdir -p %t.dir/families
// RUN: cp %S/../../oplib/level3/int_binary_elementwise_templates.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/int_tile_scalar_elementwise_templates.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/int_unary_templates.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/families/a5_oplib_v1_manifest.yaml %t.dir/families/
// RUN: { ptoas %S/bitwise_shift_family_static.pto --enable-op-fusion --op-lib-dir=%t.dir --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s --check-prefix=STATIC
// RUN: rm -rf %t.dir && mkdir -p %t.dir/families
// RUN: cp %S/../../oplib/level3/int_binary_elementwise_templates.mlir %t.dir/
// XFAIL: *
// RUN: cp %S/../../oplib/level3/int_tile_scalar_elementwise_templates.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/int_unary_templates.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/families/a5_oplib_v1_manifest.yaml %t.dir/families/
// RUN: { ptoas %S/bitwise_shift_family_dynamic_vshape.pto --enable-op-fusion --op-lib-dir=%t.dir --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s --check-prefix=DYNAMIC
// RUN: rm -rf %t.dir && mkdir -p %t.dir/families
// RUN: cp %S/../../oplib/level3/int_binary_elementwise_i16_templates.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/int_tile_scalar_elementwise_i16_templates.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/int_unary_i16_templates.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/families/a5_oplib_v1_manifest.yaml %t.dir/families/
// RUN: { ptoas %S/bitwise_shift_family_i16_smoke.pto --enable-op-fusion --op-lib-dir=%t.dir --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s --check-prefix=I16

// STATIC-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// STATIC-DAG: pto.oplib.instance.op = "tand"
// STATIC-DAG: pto.oplib.instance.op = "tor"
// STATIC-DAG: pto.oplib.instance.op = "txor"
// STATIC-DAG: pto.oplib.instance.op = "tshl"
// STATIC-DAG: pto.oplib.instance.op = "tshr"
// STATIC-DAG: pto.oplib.instance.op = "tnot"

// DYNAMIC-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// DYNAMIC-DAG: pto.oplib.instance.op = "tand"
// DYNAMIC-DAG: pto.oplib.instance.op = "tor"
// DYNAMIC-DAG: pto.oplib.instance.op = "txor"
// DYNAMIC-DAG: pto.oplib.instance.op = "tshl"
// DYNAMIC-DAG: pto.oplib.instance.op = "tshr"
// DYNAMIC-DAG: pto.oplib.instance.op = "tnot"

// I16-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// I16-DAG: pto.oplib.instance.op = "tand"
// I16-DAG: pto.oplib.instance.op = "tnot"
