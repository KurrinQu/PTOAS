// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-allow-unresolved --a5vm-unresolved-report=%t.unresolved %s -o - | FileCheck %s

// CHECK-LABEL: define void @abs_kernel_2d
// CHECK: call {{.*}}llvm.hivm
// CHECK: ; A5VM-UNRESOLVED:
module {
  func.func @abs_kernel_2d(%base: memref<1024xf32>, %out: memref<1024xf32>, %index: index) {
    %loaded = a5vm.load %base[%index] {
      layout = "nd",
      valid_rows = 32,
      valid_cols = 32,
      domain = "gm"
    } : memref<1024xf32> -> !a5vm.vec<64xf32>
    %abs = a5vm.abs %loaded : !a5vm.vec<64xf32> -> !a5vm.vec<64xf32>
    a5vm.store %abs, %out[%index] {
      layout = "nd",
      domain = "vec"
    } : !a5vm.vec<64xf32>, memref<1024xf32>
    return
  }
}
