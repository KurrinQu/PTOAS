// RUN: ptoas %s --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --print-ir-after-all --print-ir-after-all-func-filter=family_b8 -o /dev/null > %t.b8 2>&1
// RUN: awk '/IR Dump After PTOFusionPredicateElision/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t.b8 | FileCheck %s --check-prefix=B8
// RUN: ptoas %s --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --print-ir-after-all --print-ir-after-all-func-filter=family_b16 -o /dev/null > %t.b16 2>&1
// RUN: awk '/IR Dump After PTOFusionPredicateElision/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t.b16 | FileCheck %s --check-prefix=B16
// RUN: ptoas %s --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --print-ir-after-all --print-ir-after-all-func-filter=family_b32 -o /dev/null > %t.b32 2>&1
// RUN: awk '/IR Dump After PTOFusionPredicateElision/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t.b32 | FileCheck %s --check-prefix=B32
// RUN: ptoas %s --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --print-ir-after-all --print-ir-after-all-func-filter=bitwidth_negative -o /dev/null > %t.bit 2>&1
// RUN: awk '/IR Dump After PTOFusionPredicateElision/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t.bit | FileCheck %s --check-prefix=BIT
// RUN: ptoas %s --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --print-ir-after-all --print-ir-after-all-func-filter=recurrence_negative_init_mismatch -o /dev/null > %t.rec 2>&1
// RUN: awk '/IR Dump After PTOFusionPredicateElision/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t.rec | FileCheck %s --check-prefix=REC

module {
  func.func @family_b8(
      %src_raw: memref<1x257xi8, strided<[257, 1], offset: ?>, #pto.address_space<vec>>,
      %tmp_raw: memref<1x257xi8, strided<[257, 1], offset: ?>, #pto.address_space<vec>>,
      %dst_raw: memref<1x257xi8, strided<[257, 1], offset: ?>, #pto.address_space<vec>>) {
    %c1 = arith.constant 1 : index
    %c257 = arith.constant 257 : index
    %src = pto.bind_tile %src_raw, %c1, %c257 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x257xi8, strided<[257, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x257xi8, strided<[257, 1], offset: ?>, #pto.address_space<vec>>
    %tmp = pto.bind_tile %tmp_raw, %c1, %c257 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x257xi8, strided<[257, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x257xi8, strided<[257, 1], offset: ?>, #pto.address_space<vec>>
    %dst = pto.bind_tile %dst_raw, %c1, %c257 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x257xi8, strided<[257, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x257xi8, strided<[257, 1], offset: ?>, #pto.address_space<vec>>
    "pto.fusion_region"() ({
      pto.tnot ins(%src : memref<1x257xi8, strided<[257, 1], offset: ?>, #pto.address_space<vec>>) outs(%tmp : memref<1x257xi8, strided<[257, 1], offset: ?>, #pto.address_space<vec>>)
      pto.tnot ins(%tmp : memref<1x257xi8, strided<[257, 1], offset: ?>, #pto.address_space<vec>>) outs(%dst : memref<1x257xi8, strided<[257, 1], offset: ?>, #pto.address_space<vec>>)
      "pto.yield"() : () -> ()
    }) : () -> ()
    return
  }

  func.func @family_b16(
      %src_raw: memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>,
      %tmp_raw: memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>,
      %dst_raw: memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>) {
    %c1 = arith.constant 1 : index
    %c129 = arith.constant 129 : index
    %src = pto.bind_tile %src_raw, %c1, %c129 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>
    %tmp = pto.bind_tile %tmp_raw, %c1, %c129 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>
    %dst = pto.bind_tile %dst_raw, %c1, %c129 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>
    "pto.fusion_region"() ({
      pto.tnot ins(%src : memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>) outs(%tmp : memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>)
      pto.tnot ins(%tmp : memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>) outs(%dst : memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>)
      "pto.yield"() : () -> ()
    }) : () -> ()
    return
  }

  func.func @family_b32(
      %src_raw: memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>,
      %tmp_raw: memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>,
      %dst_raw: memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>) {
    %c1 = arith.constant 1 : index
    %c65 = arith.constant 65 : index
    %src = pto.bind_tile %src_raw, %c1, %c65 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>
    %tmp = pto.bind_tile %tmp_raw, %c1, %c65 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>
    %dst = pto.bind_tile %dst_raw, %c1, %c65 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>
    "pto.fusion_region"() ({
      pto.tnot ins(%src : memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>) outs(%tmp : memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>)
      pto.tnot ins(%tmp : memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>) outs(%dst : memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>)
      "pto.yield"() : () -> ()
    }) : () -> ()
    return
  }

  func.func @bitwidth_negative(
      %src16_raw: memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>,
      %dst16_raw: memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>,
      %src32_raw: memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>,
      %dst32_raw: memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>) {
    %c1 = arith.constant 1 : index
    %c129 = arith.constant 129 : index
    %c65 = arith.constant 65 : index
    %src16 = pto.bind_tile %src16_raw, %c1, %c129 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>
    %dst16 = pto.bind_tile %dst16_raw, %c1, %c129 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>
    %src32 = pto.bind_tile %src32_raw, %c1, %c65 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>
    %dst32 = pto.bind_tile %dst32_raw, %c1, %c65 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>
    "pto.fusion_region"() ({
      pto.tnot ins(%src16 : memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>) outs(%dst16 : memref<1x129xi16, strided<[129, 1], offset: ?>, #pto.address_space<vec>>)
      pto.tnot ins(%src32 : memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>) outs(%dst32 : memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>)
      "pto.yield"() : () -> ()
    }) : () -> ()
    return
  }

  func.func @recurrence_negative_init_mismatch(
      %src0_raw: memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>,
      %dst0_raw: memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>,
      %src1_raw: memref<1x66xi32, strided<[66, 1], offset: ?>, #pto.address_space<vec>>,
      %dst1_raw: memref<1x66xi32, strided<[66, 1], offset: ?>, #pto.address_space<vec>>) {
    %c1 = arith.constant 1 : index
    %c65 = arith.constant 65 : index
    %c66 = arith.constant 66 : index
    %src0 = pto.bind_tile %src0_raw, %c1, %c65 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>
    %dst0 = pto.bind_tile %dst0_raw, %c1, %c65 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>
    %src1 = pto.bind_tile %src1_raw, %c1, %c66 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x66xi32, strided<[66, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x66xi32, strided<[66, 1], offset: ?>, #pto.address_space<vec>>
    %dst1 = pto.bind_tile %dst1_raw, %c1, %c66 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x66xi32, strided<[66, 1], offset: ?>, #pto.address_space<vec>> -> memref<1x66xi32, strided<[66, 1], offset: ?>, #pto.address_space<vec>>
    "pto.fusion_region"() ({
      pto.tnot ins(%src0 : memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>) outs(%dst0 : memref<1x65xi32, strided<[65, 1], offset: ?>, #pto.address_space<vec>>)
      pto.tnot ins(%src1 : memref<1x66xi32, strided<[66, 1], offset: ?>, #pto.address_space<vec>>) outs(%dst1 : memref<1x66xi32, strided<[66, 1], offset: ?>, #pto.address_space<vec>>)
      "pto.yield"() : () -> ()
    }) : () -> ()
    return
  }
}

// B8-LABEL: func.func @family_b8(
// B8: %[[MASK8:[^,]+]], %[[SCALAR8:[^ ]+]] = a5vm.plt_b8 %{{[^ ]+}} : i32 -> !a5vm.mask, i32
// B8-NOT: a5vm.plt_b8
// B8: scf.yield %[[SCALAR8]], %[[SCALAR8]] : i32, i32

// B16-LABEL: func.func @family_b16(
// B16: %[[MASK16:[^,]+]], %[[SCALAR16:[^ ]+]] = a5vm.plt_b16 %{{[^ ]+}} : i32 -> !a5vm.mask, i32
// B16-NOT: a5vm.plt_b16
// B16: scf.yield %[[SCALAR16]], %[[SCALAR16]] : i32, i32

// B32-LABEL: func.func @family_b32(
// B32: %[[MASK32:[^,]+]], %[[SCALAR32:[^ ]+]] = a5vm.plt_b32 %{{[^ ]+}} : i32 -> !a5vm.mask, i32
// B32-NOT: a5vm.plt_b32
// B32: scf.yield %[[SCALAR32]], %[[SCALAR32]] : i32, i32

// BIT-LABEL: func.func @bitwidth_negative(
// BIT: a5vm.plt_b16 %{{[^ ]+}} : i32 -> !a5vm.mask, i32
// BIT: a5vm.plt_b32 %{{[^ ]+}} : i32 -> !a5vm.mask, i32

// REC-LABEL: func.func @recurrence_negative_init_mismatch(
// REC: %[[MASK0:[^,]+]], %[[OUT0:[^ ]+]] = a5vm.plt_b32 %{{[^ ]+}} : i32 -> !a5vm.mask, i32
// REC: scf.yield %[[OUT0]] : i32
// REC: %[[MASK1:[^,]+]], %[[OUT1:[^ ]+]] = a5vm.plt_b32 %{{[^ ]+}} : i32 -> !a5vm.mask, i32
// REC: scf.yield %[[OUT1]] : i32
