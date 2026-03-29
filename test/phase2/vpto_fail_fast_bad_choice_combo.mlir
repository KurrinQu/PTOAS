// RUN: ! ptoas --pto-backend=vpto %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'pto.tadd' op add lowering supports only f16, f32, bf16, and 8/16/32-bit integer element types
// CHECK: error: 'pto.tadd' op failed to lower with 'pto.lowering_choice' = #pto.lowering_choice<update_mode = no_post_update, loop_shape = two_d>
// CHECK: Error: VPTO backend lowering pass execution failed.

module {
  func.func @bad_choice_combo(%src0: memref<1x16xi64, strided<[16, 1]>, #pto.address_space<vec>>, %src1: memref<1x16xi64, strided<[16, 1]>, #pto.address_space<vec>>, %dst: memref<1x16xi64, strided<[16, 1]>, #pto.address_space<vec>>) attributes {pto.version_selection_applied} {
    "pto.tadd"(%src0, %src1, %dst) {pto.lowering_choice = #pto.lowering_choice<update_mode = no_post_update, loop_shape = two_d>} : (memref<1x16xi64, strided<[16, 1]>, #pto.address_space<vec>>, memref<1x16xi64, strided<[16, 1]>, #pto.address_space<vec>>, memref<1x16xi64, strided<[16, 1]>, #pto.address_space<vec>>) -> ()
    return
  }
}
