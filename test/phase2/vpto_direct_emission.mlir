// RUN: ptoas --pto-backend=vpto --print-ir-after-all --print-ir-after-all-func-filter=direct_tadd %s -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After PTOToVPTO/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s

// CHECK-LABEL: IR Dump After PTOToVPTO
// CHECK: func.func @direct_tadd
// CHECK: scf.for
// CHECK: pto.plt_b32
// CHECK: pto.vlds
// CHECK: pto.vadd
// CHECK: pto.vsts
// CHECK-NOT: scf.if
// CHECK-NOT: pto.tadd
// CHECK-NOT: pto.lowering_choice

module {
  func.func @direct_tadd(%src0_raw: memref<1x16xf32, strided<[16, 1], offset: ?>, #pto.address_space<vec>>, %src1_raw: memref<1x16xf32, strided<[16, 1], offset: ?>, #pto.address_space<vec>>, %dst_raw: memref<1x16xf32, strided<[16, 1], offset: ?>, #pto.address_space<vec>>) attributes {pto.version_selection_applied} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %src0 = pto.bind_tile %src0_raw, %c1, %c16 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x16xf32, strided<[16, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x16xf32, strided<[16, 1], offset: ?>, #pto.address_space<vec>>
    %src1 = pto.bind_tile %src1_raw, %c1, %c16 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x16xf32, strided<[16, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x16xf32, strided<[16, 1], offset: ?>, #pto.address_space<vec>>
    %dst = pto.bind_tile %dst_raw, %c1, %c16 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x16xf32, strided<[16, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x16xf32, strided<[16, 1], offset: ?>, #pto.address_space<vec>>
    pto.tadd ins(%src0, %src1 : memref<1x16xf32, strided<[16, 1], offset: ?>, #pto.address_space<vec>>, memref<1x16xf32, strided<[16, 1], offset: ?>, #pto.address_space<vec>>) outs(%dst : memref<1x16xf32, strided<[16, 1], offset: ?>, #pto.address_space<vec>>) {pto.lowering_choice = #pto.lowering_choice<update_mode = no_post_update, loop_shape = two_d>}
    return
  }
}
