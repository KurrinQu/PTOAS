// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto %s -o - 2>/dev/null | FileCheck %s

// CHECK-LABEL: func.func @memref_subview_vld_vst_boundary(
// CHECK-SAME: %arg0: !pto.ptr<f32, ub>, %arg1: !pto.ptr<f32, ub>, %arg2: index, %arg3: !pto.mask<b32>
// CHECK-NOT: memref.subview
// CHECK: pto.vecscope {
// CHECK: arith.muli %arg2, %{{.+}} : index
// CHECK: %[[SRC_BASE:.+]] = pto.addptr %arg0, %{{.+}} : <f32, ub> -> <f32, ub>
// CHECK: %[[LOAD:.+]] = pto.vlds %[[SRC_BASE]][%{{.+}}] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
// CHECK: arith.muli %arg2, %{{.+}} : index
// CHECK: %[[DST_BASE:.+]] = pto.addptr %arg1, %{{.+}} : <f32, ub> -> <f32, ub>
// CHECK: pto.vsts %[[LOAD]], %[[DST_BASE]][%{{.+}}], %arg3 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

module {
  func.func @memref_subview_vld_vst_boundary(
      %src: memref<8x64xf32, #pto.address_space<vec>>,
      %dst: memref<8x64xf32, #pto.address_space<vec>>,
      %row: index, %mask: !pto.mask<b32>)
      attributes {pto.version_selection_applied} {
    %c0 = arith.constant 0 : index
    %src_row = memref.subview %src[%row, 0] [1, 64] [1, 1]
      : memref<8x64xf32, #pto.address_space<vec>>
         to memref<1x64xf32, strided<[64, 1], offset: ?>, #pto.address_space<vec>>
    %dst_row = memref.subview %dst[%row, 0] [1, 64] [1, 1]
      : memref<8x64xf32, #pto.address_space<vec>>
         to memref<1x64xf32, strided<[64, 1], offset: ?>, #pto.address_space<vec>>
    pto.vecscope {
      %v = pto.vlds %src_row[%c0, %c0] : memref<1x64xf32, strided<[64, 1], offset: ?>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
      pto.vsts %v, %dst_row[%c0, %c0], %mask : !pto.vreg<64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #pto.address_space<vec>>, !pto.mask<b32>
    }
    return
  }
}
