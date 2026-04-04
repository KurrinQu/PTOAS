// RUN: ! ptoas --pto-backend=vpto --emit-vpto %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'pto.vlds' op requires either one linearized index or one index per memref dimension for source; got 3 indices for rank 2
// CHECK: Error: Failed to parse MLIR.

module {
  func.func @memref_multidim_rank_mismatch(
      %src: memref<8x64xf32, #pto.address_space<vec>>, %i: index, %j: index,
      %k: index) attributes {pto.version_selection_applied} {
    pto.vecscope {
      %v = pto.vlds %src[%i, %j, %k] : memref<8x64xf32, #pto.address_space<vec>> -> !pto.vreg<64xf32>
    }
    return
  }
}
