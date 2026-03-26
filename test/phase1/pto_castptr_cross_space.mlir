// RUN: ./build/tools/ptoas/ptoas %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @pto_castptr_cross_space(%arg0: !pto.ptr<f32>) {
    // CHECK: error: 'pto.castptr' op ptr-to-ptr cast must stay within the same PTO memory space
    %0 = pto.castptr %arg0 : !pto.ptr<f32> -> !pto.ptr<i8, ub>
    return
  }
}
