// RUN: ! ptoas --pto-backend=a5vm %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'pto.tadd' op requires 'pto.a5vm_lowering_choice' before PTOToA5VM in A5 fusion mainline, but the attribute is missing
// CHECK: Error: A5VM backend lowering pass execution failed.

module {
  func.func @missing_choice(%src0: memref<1x16xf32, strided<[16, 1]>, #pto.address_space<vec>>, %src1: memref<1x16xf32, strided<[16, 1]>, #pto.address_space<vec>>, %dst: memref<1x16xf32, strided<[16, 1]>, #pto.address_space<vec>>) attributes {pto.a5vm_version_selection_applied} {
    pto.tadd ins(%src0, %src1 : memref<1x16xf32, strided<[16, 1]>, #pto.address_space<vec>>, memref<1x16xf32, strided<[16, 1]>, #pto.address_space<vec>>) outs(%dst : memref<1x16xf32, strided<[16, 1]>, #pto.address_space<vec>>)
    return
  }
}
