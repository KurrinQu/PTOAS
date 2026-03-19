// RUN: ptoas %s --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --pto-level=level3 -o %t.cpp
// RUN: FileCheck %s --check-prefix=EMITC < %t.cpp

// EMITC-LABEL: __global__ AICORE void compare_bitwise_family_emitc(
// EMITC-DAG: __VEC_SCOPE__ {
// EMITC-DAG: CreatePredicate<float>(
// EMITC-DAG: vlds(
// EMITC-DAG: vsts(
// EMITC-DAG: vcmp(
// EMITC-DAG: LT
// EMITC-DAG: vsel(
// EMITC-DAG: vand(
// EMITC-DAG: vshrs(
// EMITC-DAG: vxor(

module {
  func.func @compare_bitwise_family_emitc(
      %lhs: memref<1024xf32, strided<[1], offset: 0>, #pto.address_space<vec>>,
      %rhs: memref<1024xf32, strided<[1], offset: 0>, #pto.address_space<vec>>,
      %dst_sel: memref<1024xf32, strided<[1], offset: 0>, #pto.address_space<vec>>,
      %dst_cmp: memref<1024xi8, strided<[1], offset: 0>, #pto.address_space<vec>>,
      %bits0: memref<1024xi32, strided<[1], offset: 0>, #pto.address_space<vec>>,
      %bits1: memref<1024xi32, strided<[1], offset: 0>, #pto.address_space<vec>>,
      %dst_bits: memref<1024xi32, strided<[1], offset: 0>, #pto.address_space<vec>>)
      attributes {pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %passive = arith.constant dense<0.0> : vector<64xf32>
    %zero_i8 = arith.constant dense<0> : vector<64xi8>
    %one_i8 = arith.constant dense<1> : vector<64xi8>
    %allones = arith.constant dense<-1> : vector<64xi32>

    pto.simd.vec_scope {
      %active = vector.create_mask %c64 : vector<64xi1>
      %lhs_v = vector.maskedload %lhs[%c0], %active, %passive {pto.simd.vld_dist = "NORM"} : memref<1024xf32, strided<[1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
      %rhs_v = vector.maskedload %rhs[%c0], %active, %passive {pto.simd.vld_dist = "NORM"} : memref<1024xf32, strided<[1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
      %cmp = arith.cmpf olt, %lhs_v, %rhs_v : vector<64xf32>
      %sel = arith.select %cmp, %lhs_v, %rhs_v : vector<64xi1>, vector<64xf32>
      vector.maskedstore %dst_sel[%c0], %active, %sel {pto.simd.vst_dist = "DIST_NORM"} : memref<1024xf32, strided<[1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
      vector.maskedstore %dst_cmp[%c0], %active, %zero_i8 {pto.simd.vst_dist = "DIST_NORM"} : memref<1024xi8, strided<[1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
      vector.maskedstore %dst_cmp[%c0], %cmp, %one_i8 {pto.simd.vst_dist = "DIST_NORM"} : memref<1024xi8, strided<[1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>

      %bits0_v = vector.load %bits0[%c0] {pto.simd.vld_dist = "NORM"} : memref<1024xi32, strided<[1], offset: 0>, #pto.address_space<vec>>, vector<64xi32>
      %bits1_v = vector.load %bits1[%c0] {pto.simd.vld_dist = "NORM"} : memref<1024xi32, strided<[1], offset: 0>, #pto.address_space<vec>>, vector<64xi32>
      %anded = arith.andi %bits0_v, %bits1_v : vector<64xi32>
      %shifted = arith.shrsi %anded, %bits1_v : vector<64xi32>
      %not_v = arith.xori %shifted, %allones : vector<64xi32>
      vector.store %not_v, %dst_bits[%c0] {pto.simd.vst_dist = "DIST_NORM"} : memref<1024xi32, strided<[1], offset: 0>, #pto.address_space<vec>>, vector<64xi32>
    }
    return
  }
}
