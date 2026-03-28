// RUN: ptoas %s --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --pto-level=level3 -o %t.cpp
// RUN: FileCheck %s --check-prefix=EMITC < %t.cpp

// EMITC-LABEL: __global__ AICORE void reduction_broadcast_family_emitc_f32(
// EMITC-DAG: __VEC_SCOPE__ {
// EMITC-DAG: CreatePredicate<float>(
// EMITC-DAG: vlds(
// EMITC-DAG: vsts(
// EMITC-DAG: vdup(
// EMITC-DAG: vmul(
// EMITC-DAG: ptoas_vreduce_add(

module {
  func.func @reduction_broadcast_family_emitc_f32(
      %src: memref<1024xf32, strided<[1], offset: 0>, #pto.address_space<vec>>,
      %dst: memref<1024xf32, strided<[1], offset: 0>, #pto.address_space<vec>>,
      %scalar: f32)
      attributes {pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} {
    %c0 = arith.constant 0 : index

    pto.simd.vec_scope {
      %src_v = vector.load %src[%c0] {pto.simd.vld_dist = "NORM"} : memref<1024xf32, strided<[1], offset: 0>, #pto.address_space<vec>>, vector<64xf32>
      %dup = vector.splat %scalar : vector<64xf32>
      %mul = arith.mulf %src_v, %dup {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>
      %reduced = vector.reduction <add>, %mul, %scalar : vector<64xf32> into f32
      %broadcast = vector.splat %reduced : vector<64xf32>
      vector.store %broadcast, %dst[%c0] {pto.simd.vst_dist = "DIST_NORM"} : memref<1024xf32, strided<[1], offset: 0>, #pto.address_space<vec>>, vector<64xf32>
    }
    return
  }
}
