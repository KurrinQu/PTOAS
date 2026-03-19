// RUN: FileCheck %s --check-prefix=FBIN < %S/../../oplib/level3/float_binary_elementwise_templates.mlir
// RUN: FileCheck %s --check-prefix=IBIN-U8 < %S/../../oplib/level3/int_binary_elementwise_i8_templates.mlir
// RUN: FileCheck %s --check-prefix=IBIN-U16 < %S/../../oplib/level3/int_binary_elementwise_i16_templates.mlir
// RUN: FileCheck %s --check-prefix=IBIN-U32 < %S/../../oplib/level3/int_binary_elementwise_templates.mlir
// RUN: FileCheck %s --check-prefix=PBIN < %S/../../oplib/level3/float_partial_binary_templates.mlir
// RUN: FileCheck %s --check-prefix=FTS < %S/../../oplib/level3/float_tile_scalar_templates.mlir
// RUN: FileCheck %s --check-prefix=ITS-U8 < %S/../../oplib/level3/int_tile_scalar_elementwise_i8_templates.mlir
// RUN: FileCheck %s --check-prefix=ITS-U16 < %S/../../oplib/level3/int_tile_scalar_elementwise_i16_templates.mlir
// RUN: FileCheck %s --check-prefix=ITS-U32 < %S/../../oplib/level3/int_tile_scalar_elementwise_templates.mlir
// RUN: FileCheck %s --check-prefix=IUN-U8 < %S/../../oplib/level3/int_unary_i8_templates.mlir
// RUN: FileCheck %s --check-prefix=IUN-U16 < %S/../../oplib/level3/int_unary_i16_templates.mlir
// RUN: FileCheck %s --check-prefix=IUN-U32 < %S/../../oplib/level3/int_unary_templates.mlir

// FBIN: func.func private @__pto_oplib_variant_tadd_bf16(
// FBIN: pto.oplib.match.dtype = "bf16"
// FBIN: func.func private @__pto_oplib_variant_tdiv_bf16(

// IBIN-U8: // axes = dtype=u8, core_op=arith.divui, variant_id=tile_u8
// IBIN-U8: func.func private @__pto_oplib_variant_tdiv_u8(
// IBIN-U8: %result = arith.divui %lhs, %rhs : vector<256xi8>

// IBIN-U16: // axes = dtype=u16, core_op=arith.remui, variant_id=tile_u16
// IBIN-U16: func.func private @__pto_oplib_variant_trem_u16(
// IBIN-U16: %result = arith.remui %lhs, %rhs : vector<128xi16>

// IBIN-U32: // axes = dtype=u32, core_op=arith.maxui, variant_id=tile_u32
// IBIN-U32: func.func private @__pto_oplib_variant_tmax_u32(
// IBIN-U32: %result = arith.maxui %lhs, %rhs : vector<64xi32>
// IBIN-U32: // axes = dtype=u32, core_op=arith.minui, variant_id=tile_u32
// IBIN-U32: func.func private @__pto_oplib_variant_tmin_u32(

// PBIN: func.func private @__pto_oplib_variant_tpartadd_u32(
// PBIN: func.func private @__pto_oplib_variant_tpartadd_bf16(
// PBIN: // axes = dtype=u32, core_op=arith.maxui, variant_id=tile_u32
// PBIN: func.func private @__pto_oplib_variant_tpartmax_u32(
// PBIN: %result = arith.maxui %lhs, %rhs : vector<64xi32>
// PBIN: func.func private @__pto_oplib_variant_tprelu_f16(
// PBIN: func.func private @__pto_oplib_variant_tprelu_f32(
// PBIN-NOT: __pto_oplib_variant_tprelu_bf16(

// FTS: func.func private @__pto_oplib_variant_tadds_bf16(
// FTS: func.func private @__pto_oplib_variant_tmuls_bf16(
// FTS: func.func private @__pto_oplib_variant_tdivs_tile_scalar_f16_f16(
// FTS: func.func private @__pto_oplib_variant_tdivs_scalar_tile_f16_f16(
// FTS: func.func private @__pto_oplib_variant_tlrelu_f16(
// FTS: func.func private @__pto_oplib_variant_tlrelu_f32(
// FTS-NOT: __pto_oplib_variant_tdivs_tile_scalar_bf16_bf16(
// FTS-NOT: __pto_oplib_variant_tdivs_scalar_tile_bf16_bf16(
// FTS-NOT: __pto_oplib_variant_tlrelu_bf16(

// ITS-U8: func.func private @__pto_oplib_variant_tdivs_tile_scalar_u8_u8(
// ITS-U8: %result = arith.divui %lhs, %scalarVec : vector<256xi8>
// ITS-U8: func.func private @__pto_oplib_variant_tdivs_scalar_tile_u8_u8(
// ITS-U8: %result = arith.divui %scalarVec, %lhs : vector<256xi8>

// ITS-U16: func.func private @__pto_oplib_variant_trems_u16(
// ITS-U16: %result = arith.remui %lhs, %scalarVec : vector<128xi16>

// ITS-U32: func.func private @__pto_oplib_variant_tmaxs_u32(
// ITS-U32: %result = arith.maxui %lhs, %scalarVec : vector<64xi32>
// ITS-U32: func.func private @__pto_oplib_variant_tmins_u32(
// ITS-U32: %result = arith.minui %lhs, %scalarVec : vector<64xi32>
// ITS-U32: func.func private @__pto_oplib_variant_trems_u32(

// IUN-U8: func.func private @__pto_oplib_variant_tnot_u8(
// IUN-U8: pto.oplib.match.dtype = "u8"

// IUN-U16: func.func private @__pto_oplib_variant_tnot_u16(
// IUN-U16: pto.oplib.match.dtype = "u16"

// IUN-U32: func.func private @__pto_oplib_variant_tnot_u32(
// IUN-U32: pto.oplib.match.dtype = "u32"
