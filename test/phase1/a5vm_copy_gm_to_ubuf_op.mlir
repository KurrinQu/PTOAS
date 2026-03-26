// RUN: ./build/tools/ptoas/ptoas %s -o - | FileCheck %s

// CHECK-LABEL: @copy_gm_to_ubuf
// CHECK: a5vm.copy_gm_to_ubuf %arg0, %arg1, %[[ROWS:[^,]+]], %[[COLS:[^,]+]], %[[ZERO:[^,]+]], %[[NBURST:[^,]+]], %[[LEN:[^,]+]], %[[ZERO]], %[[ZERO]], %[[ZERO]], %[[GMSTRIDE:[^,]+]], %[[UBSTRIDE:[^ ]+]]
// CHECK-SAME: {data_select_bit = false, layout = "nd", ub_pad = false}
// CHECK-SAME: : !pto.ptr<i8, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
module {
  func.func @copy_gm_to_ubuf(%src: !pto.ptr<i8, gm>, %dst: !pto.ptr<f32, ub>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    a5vm.copy_gm_to_ubuf %src, %dst, %c32_i64, %c32_i64, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c0_i64, %c0_i64, %c128_i64, %c128_i64 {layout = "nd", data_select_bit = false, ub_pad = false} : !pto.ptr<i8, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
    return
  }
}

// CHECK: error: 'a5vm.copy_gm_to_ubuf' op requires GM source, UB destination, and complete transfer metadata
module {
  func.func @copy_gm_to_ubuf_missing_metadata(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<i8, gm>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    a5vm.copy_gm_to_ubuf %src, %dst, %c32_i64, %c32_i64, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c0_i64, %c0_i64, %c128_i64, %c128_i64 {layout = "nd", ub_pad = false} : !pto.ptr<f32, ub>, !pto.ptr<i8, gm>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64
    return
  }
}
