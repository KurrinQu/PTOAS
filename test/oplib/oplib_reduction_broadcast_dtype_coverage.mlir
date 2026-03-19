// RUN: FileCheck %s --check-prefix=ROW < %S/../../oplib/level3/reduce_row_templates.mlir
// RUN: FileCheck %s --check-prefix=COL < %S/../../oplib/level3/reduce_col_templates.mlir
// RUN: FileCheck %s --check-prefix=COLSUM < %S/../../oplib/level3/reduce_colsum_templates.mlir
// RUN: FileCheck %s --check-prefix=BROW < %S/../../oplib/level3/broadcast_row_templates.mlir
// RUN: FileCheck %s --check-prefix=BCOL < %S/../../oplib/level3/broadcast_col_templates.mlir
// RUN: FileCheck %s --check-prefix=BROWBIN < %S/../../oplib/level3/broadcast_row_binary_templates.mlir
// RUN: FileCheck %s --check-prefix=SEXPAND < %S/../../oplib/level3/scalar_expand_templates.mlir

// ROW: // axes = dtype=f16, core_op=arith.addf, variant_id=linear
// ROW: func.func private @__pto_oplib_variant_trowsum_linear_f16(
// ROW: // axes = dtype=f16, core_op=arith.maximumf, variant_id=linear
// ROW: func.func private @__pto_oplib_variant_trowmax_linear_f16(

// COL: // axes = dtype=i32, core_op=arith.maxsi, variant_id=linear
// COL: func.func private @__pto_oplib_variant_tcolmax_linear_i32(
// COL: // axes = dtype=u8, core_op=arith.minui, variant_id=linear
// COL: func.func private @__pto_oplib_variant_tcolmin_linear_u8(

// COLSUM: // axes = dtype=bf16, core_op=arith.addf, variant_id=linear
// COLSUM: func.func private @__pto_oplib_variant_tcolsum_linear_bf16(
// COLSUM: // axes = dtype=u16, core_op=arith.addi, variant_id=linear
// COLSUM: func.func private @__pto_oplib_variant_tcolsum_linear_u16(

// BROW: // axes = dtype=u16, core_op=arith.addi, variant_id=linear
// BROW: func.func private @__pto_oplib_variant_trowexpand_linear_u16(

// BCOL-DAG: // axes = dtype=i8, core_op=arith.addi, variant_id=linear
// BCOL-DAG: func.func private @__pto_oplib_variant_tcolexpand_linear_i8(
// BCOL-DAG: // axes = dtype=bf16, core_op=arith.addf, variant_id=linear
// BCOL-DAG: func.func private @__pto_oplib_variant_tcolexpand_linear_bf16(

// BROWBIN: // axes = dtype=f16, core_op=arith.mulf, variant_id=linear
// BROWBIN: func.func private @__pto_oplib_variant_trowexpandmul_linear_f16(
// BROWBIN-NOT: variant_trowexpandmul_linear_bf16

// SEXPAND: // axes = dtype=u32, core_op=arith.addi, variant_id=scalar
// SEXPAND: func.func private @__pto_oplib_variant_texpands_scalar_u32(
// SEXPAND: pto.oplib.match.dtype = "u32"
