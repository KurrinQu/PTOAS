// RUN: FileCheck %s --check-prefix=CMP-TILE < %S/../../oplib/level3/cmp_tile_tile_templates.mlir
// RUN: FileCheck %s --check-prefix=CMP-SCALAR < %S/../../oplib/level3/cmp_tile_scalar_templates.mlir
// RUN: FileCheck %s --check-prefix=SEL-MASK < %S/../../oplib/level3/select_mask_templates.mlir

// CMP-TILE-LABEL: func.func private @__pto_oplib_variant_tcmp_ge_f16(
// CMP-TILE: %mask = vector.create_mask %active : vector<128xi1>
// CMP-TILE: %linearBase = arith.muli %r, %cols : index
// CMP-TILE: %linear = arith.addi %linearBase, %cidx : index
// CMP-TILE: pto.simd.store_predicate %lhs, %src1v, %dst, %linear, %active
// CMP-TILE-NOT: vector.maskedstore

// CMP-SCALAR-LABEL: func.func private @__pto_oplib_variant_tcmps_ge_i16(
// CMP-SCALAR: %mask = vector.create_mask %active : vector<128xi1>
// CMP-SCALAR: %linearBase = arith.muli %r, %cols : index
// CMP-SCALAR: %linear = arith.addi %linearBase, %cidx : index
// CMP-SCALAR: pto.simd.store_predicate %lhs, %scalar, %dst, %linear, %active
// CMP-SCALAR-NOT: vector.maskedstore

// SEL-MASK-LABEL: func.func private @__pto_oplib_variant_tsel_mask_i16(
// SEL-MASK: %passiveMask = arith.constant dense<0> : vector<128xi8>
// SEL-MASK: %zeroMask = arith.constant dense<0> : vector<128xi8>
// SEL-MASK: %laneMask = vector.create_mask %active : vector<128xi1>
// SEL-MASK: %maskBytes = vector.maskedload %m0[%r, %cidx], %laneMask, %passiveMask
// SEL-MASK: %maskVec = arith.cmpi ne, %maskBytes, %zeroMask : vector<128xi8>
// SEL-MASK: %result = arith.select %maskVec, %lhs, %rhs
