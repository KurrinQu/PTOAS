// RUN: ./build/tools/ptoas/ptoas %s -o - | FileCheck %s

// CHECK-LABEL: @load_kernel
// CHECK: %[[LOAD:.+]] = a5vm.load %[[BASE:.+]][%[[INDEX:.+]]] {layout = "nd", valid_rows = 32, valid_cols = 32, domain = "gm"} : memref<1024xf32> -> !a5vm.vec<64xf32>
module {
  func.func @load_kernel(%base: memref<1024xf32>, %index: index) -> !a5vm.vec<64xf32> {
    %0 = a5vm.load %base[%index] {
      layout = "nd",
      valid_rows = 32,
      valid_cols = 32,
      domain = "gm"
    } : memref<1024xf32> -> !a5vm.vec<64xf32>
    return %0 : !a5vm.vec<64xf32>
  }
}
