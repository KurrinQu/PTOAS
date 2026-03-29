// RUN: ./build/tools/ptoas/ptoas --pto-backend=vpto --pto-arch a5 --emit-vpto %s -o - 2>/dev/null | FileCheck %s --check-prefix=CHECK-IR
// RUN: ./build/tools/ptoas/ptoas --pto-backend=vpto --pto-arch a5 --vpto-emit-hivm-llvm %s -o - 2>/dev/null | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-IR-LABEL: func.func @pto_castptr
// CHECK-IR: %[[P0:.*]] = pto.castptr %arg0 : i64 -> !pto.ptr<f32, ub>
// CHECK-IR: %[[P1:.*]] = pto.castptr %[[P0]] : !pto.ptr<f32, ub> -> !pto.ptr<i8, ub>
// CHECK-IR: %[[ADDR:.*]] = pto.castptr %[[P1]] : !pto.ptr<i8, ub> -> i64
// CHECK-IR: return %[[ADDR]] : i64

// CHECK-LLVM-LABEL: define{{.*}} i64 @pto_castptr
// CHECK-LLVM: inttoptr i64 %{{.*}} to ptr addrspace(6)
// CHECK-LLVM: ptrtoint ptr addrspace(6) %{{.*}} to i64

module {
  func.func @pto_castptr(%addr: i64) -> i64 {
    %0 = pto.castptr %addr : i64 -> !pto.ptr<f32, ub>
    %1 = pto.castptr %0 : !pto.ptr<f32, ub> -> !pto.ptr<i8, ub>
    %2 = pto.castptr %1 : !pto.ptr<i8, ub> -> i64
    return %2 : i64
  }
}
