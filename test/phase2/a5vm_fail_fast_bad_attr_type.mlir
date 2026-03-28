// RUN: ! ptoas --pto-backend=a5vm %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'pto.tadd' op expects 'pto.a5vm_lowering_choice' to be #pto.a5vm_lowering_choice<...>, but got "bad"
// CHECK: Error: A5VM backend lowering pass execution failed.

module {
  func.func @bad_choice_type(%src0: memref<1x16xf32, strided<[16, 1]>, #pto.address_space<vec>>, %src1: memref<1x16xf32, strided<[16, 1]>, #pto.address_space<vec>>, %dst: memref<1x16xf32, strided<[16, 1]>, #pto.address_space<vec>>) attributes {pto.a5vm_version_selection_applied} {
    "pto.tadd"(%src0, %src1, %dst) {pto.a5vm_lowering_choice = "bad"} : (memref<1x16xf32, strided<[16, 1]>, #pto.address_space<vec>>, memref<1x16xf32, strided<[16, 1]>, #pto.address_space<vec>>, memref<1x16xf32, strided<[16, 1]>, #pto.address_space<vec>>) -> ()
    return
  }
}
