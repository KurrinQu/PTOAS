// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto %s -o - 2>/dev/null | FileCheck %s --check-prefix=IR
// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-text %s -o /dev/null
// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-llvm %s -o /dev/null

// IR-LABEL: func.func @memref_multidim_boundary(
// IR-SAME: %arg0: !pto.ptr<f32, ub>
// IR-SAME: %arg1: !pto.ptr<f32, ub>
// IR-SAME: %arg2: index
// IR-SAME: %arg3: index
// IR-SAME: %arg4: !pto.mask<b32>
// IR: %[[C64:.+]] = arith.constant 64 : index
// IR: %[[LOAD_ROW:.+]] = arith.muli %arg2, %[[C64]] : index
// IR: %[[LOAD_IDX:.+]] = arith.addi %[[LOAD_ROW]], %arg3 : index
// IR: %[[LOAD:.+]] = pto.vlds %arg0[%[[LOAD_IDX]]] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
// IR: %[[C64_2:.+]] = arith.constant 64 : index
// IR: %[[STORE_ROW:.+]] = arith.muli %arg2, %[[C64_2]] : index
// IR: %[[STORE_IDX:.+]] = arith.addi %[[STORE_ROW]], %arg3 : index
// IR: pto.vsts %[[LOAD]], %arg1[%[[STORE_IDX]]], %arg4 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
// IR-NOT: pto.castptr %arg0
// IR-NOT: pto.castptr %arg1

module {
  func.func @memref_multidim_boundary(
      %src: memref<8x64xf32, #pto.address_space<vec>>,
      %dst: memref<8x64xf32, #pto.address_space<vec>>,
      %row: index, %col: index, %mask: !pto.mask<b32>)
      attributes {pto.version_selection_applied} {
    pto.vecscope {
      %v = pto.vlds %src[%row, %col] : memref<8x64xf32, #pto.address_space<vec>> -> !pto.vreg<64xf32>
      pto.vsts %v, %dst[%row, %col], %mask : !pto.vreg<64xf32>, memref<8x64xf32, #pto.address_space<vec>>, !pto.mask<b32>
    }
    return
  }
}
