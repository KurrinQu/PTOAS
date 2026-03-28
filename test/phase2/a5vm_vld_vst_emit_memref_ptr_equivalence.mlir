// RUN: ptoas --pto-arch=a5 --pto-backend=a5vm --a5vm-emit-hivm-text %s -o - 2>/dev/null | FileCheck --check-prefix=TEXT %s
// RUN: ptoas --pto-arch=a5 --pto-backend=a5vm --a5vm-emit-hivm-llvm %s -o - 2>/dev/null | FileCheck --check-prefix=LLVM %s

// TEXT-LABEL: define void @memref_form(
// TEXT: call <64 x float> @llvm.hivm.vldsx1(ptr addrspace(6)
// TEXT: call void @llvm.hivm.vstsx1(<64 x float>
// TEXT-LABEL: define void @ptr_form(
// TEXT: call <64 x float> @llvm.hivm.vldsx1(ptr addrspace(6)
// TEXT: call void @llvm.hivm.vstsx1(<64 x float>

// LLVM-LABEL: define{{.*}} @ptr_form(
// LLVM: call <64 x float> @llvm.hivm.vldsx1(ptr addrspace(6)
// LLVM: call void @llvm.hivm.vstsx1(<64 x float>
// LLVM-LABEL: define{{.*}} @memref_form(
// LLVM: %[[PTR_AS_I64_0:.*]] = ptrtoint ptr addrspace(6) %{{.*}} to i64
// LLVM: %[[I64_AS_PTR_0:.*]] = inttoptr i64 %[[PTR_AS_I64_0]] to ptr addrspace(6)
// LLVM: call <64 x float> @llvm.hivm.vldsx1(ptr addrspace(6)
// LLVM: %[[PTR_AS_I64_1:.*]] = ptrtoint ptr addrspace(6) %{{.*}} to i64
// LLVM: %[[I64_AS_PTR_1:.*]] = inttoptr i64 %[[PTR_AS_I64_1]] to ptr addrspace(6)
// LLVM: call void @llvm.hivm.vstsx1(<64 x float>

module {
  func.func @memref_form(
      %src: memref<256xf32, #pto.address_space<vec>>,
      %dst: memref<256xf32, #pto.address_space<vec>>,
      %offset: index, %mask: !a5vm.mask) {
    %v = a5vm.vlds %src[%offset] : memref<256xf32, #pto.address_space<vec>> -> !a5vm.vec<64xf32>
    a5vm.vsts %v, %dst[%offset], %mask : !a5vm.vec<64xf32>, memref<256xf32, #pto.address_space<vec>>, !a5vm.mask
    return
  }

  func.func @ptr_form(
      %src: !llvm.ptr<6>, %dst: !llvm.ptr<6>,
      %offset: index, %mask: !a5vm.mask) {
    %v = a5vm.vlds %src[%offset] : !llvm.ptr<6> -> !a5vm.vec<64xf32>
    a5vm.vsts %v, %dst[%offset], %mask : !a5vm.vec<64xf32>, !llvm.ptr<6>, !a5vm.mask
    return
  }
}
