// RUN: FileCheck %s --check-prefix=CMP-TILE-U8 < %S/../../oplib/level3/cmp_tile_tile_templates.mlir
// RUN: FileCheck %s --check-prefix=CMP-TILE-F16 < %S/../../oplib/level3/cmp_tile_tile_templates.mlir
// RUN: FileCheck %s --check-prefix=CMP-SCALAR-I16 < %S/../../oplib/level3/cmp_tile_scalar_templates.mlir
// RUN: FileCheck %s --check-prefix=CMP-SCALAR-U32 < %S/../../oplib/level3/cmp_tile_scalar_templates.mlir
// RUN: FileCheck %s --check-prefix=SEL-MASK-U8 < %S/../../oplib/level3/select_mask_templates.mlir
// RUN: FileCheck %s --check-prefix=SEL-MASK-I16 < %S/../../oplib/level3/select_mask_templates.mlir
// RUN: FileCheck %s --check-prefix=SEL-SCALAR-U8 < %S/../../oplib/level3/select_scalar_templates.mlir
// RUN: FileCheck %s --check-prefix=SEL-SCALAR-F16 < %S/../../oplib/level3/select_scalar_templates.mlir

// CMP-TILE-U8-LABEL: func.func private @__pto_oplib_variant_tcmp_lt_u8(
// CMP-TILE-U8: pto.oplib.match.dtype = "u8"
// CMP-TILE-U8: pto.oplib.match.cmp_mode = "LT"
// CMP-TILE-U8: pto.simd.lanes = 256 : i64
// CMP-TILE-U8: pto.simd.store_predicate %lhs, %src1v, %dst, %linear, %active {cmpMode = #pto<cmp lt>} : vector<256xi8>, vector<256xi8>, !pto.tile_buf

// CMP-TILE-F16-LABEL: func.func private @__pto_oplib_variant_tcmp_ge_f16(
// CMP-TILE-F16: pto.oplib.match.dtype = "f16"
// CMP-TILE-F16: pto.oplib.match.cmp_mode = "GE"
// CMP-TILE-F16: pto.simd.lanes = 128 : i64
// CMP-TILE-F16: pto.simd.store_predicate %lhs, %src1v, %dst, %linear, %active {cmpMode = #pto<cmp ge>} : vector<128xf16>, vector<128xf16>, !pto.tile_buf

// CMP-SCALAR-I16-LABEL: func.func private @__pto_oplib_variant_tcmps_ge_i16(
// CMP-SCALAR-I16: pto.oplib.match.dtype = "i16"
// CMP-SCALAR-I16: pto.oplib.match.cmp_mode = "GE"
// CMP-SCALAR-I16: pto.simd.lanes = 128 : i64
// CMP-SCALAR-I16: pto.simd.store_predicate %lhs, %scalar, %dst, %linear, %active {cmpMode = #pto<cmp ge>} : vector<128xi16>, i16, !pto.tile_buf

// CMP-SCALAR-U32-LABEL: func.func private @__pto_oplib_variant_tcmps_lt_u32(
// CMP-SCALAR-U32: pto.oplib.match.dtype = "u32"
// CMP-SCALAR-U32: pto.oplib.match.cmp_mode = "LT"
// CMP-SCALAR-U32: pto.simd.store_predicate %lhs, %scalar, %dst, %linear, %active {cmpMode = #pto<cmp lt>} : vector<64xi32>, i32, !pto.tile_buf

// SEL-MASK-U8-LABEL: func.func private @__pto_oplib_variant_tsel_mask_u8(
// SEL-MASK-U8: pto.oplib.match.dtype = "u8"
// SEL-MASK-U8: %result = arith.select %maskVec, %lhs, %rhs : vector<256xi1>, vector<256xi8>

// SEL-MASK-I16-LABEL: func.func private @__pto_oplib_variant_tsel_mask_i16(
// SEL-MASK-I16: pto.oplib.match.dtype = "i16"
// SEL-MASK-I16: %result = arith.select %maskVec, %lhs, %rhs : vector<128xi1>, vector<128xi16>

// SEL-SCALAR-U8-LABEL: func.func private @__pto_oplib_variant_tsels_scalar_mode_u8(
// SEL-SCALAR-U8: pto.oplib.match.dtype = "u8"
// SEL-SCALAR-U8: %result = arith.select %scalarVec, %lhs, %rhs : vector<256xi1>, vector<256xi8>

// SEL-SCALAR-F16-LABEL: func.func private @__pto_oplib_variant_tsels_scalar_mode_f16(
// SEL-SCALAR-F16: pto.oplib.match.dtype = "f16"
// SEL-SCALAR-F16: %result = arith.select %scalarVec, %lhs, %rhs : vector<128xi1>, vector<128xf16>
