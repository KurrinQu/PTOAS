// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-llvm %s -o - 2>/dev/null | FileCheck %s

// CHECK-LABEL: define void @two_vecscope_loops()
// CHECK: br label %{{.*}}, !llvm.loop ![[LOOP0:[0-9]+]]
// CHECK: br label %{{.*}}, !llvm.loop ![[LOOP1:[0-9]+]]
// CHECK-DAG: ![[SCOPE:[0-9]+]] = !{!"llvm.loop.aivector_scope"}
// CHECK-DAG: ![[LOOP0]] = distinct !{![[LOOP0]], ![[SCOPE]]}
// CHECK-DAG: ![[LOOP1]] = distinct !{![[LOOP1]], ![[SCOPE]]}

module {
  func.func @two_vecscope_loops() attributes {pto.version_selection_applied} {
    %c16_i32 = arith.constant 16 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %c8320_i64 = arith.constant 8320 : i64

    pto.vecscope {
      scf.for %i = %c0 to %c1 step %c1 {
        %mask, %next = pto.plt_b32 %c16_i32 : i32 -> !pto.mask, i32
        %lhs = pto.castptr %c8320_i64 : i64 -> !pto.ptr<f32, ub>
        %lhs_vec = pto.vlds %lhs[%i] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
        %rhs = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>
        %rhs_vec = pto.vlds %rhs[%i] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
        %max = pto.vmax %lhs_vec, %rhs_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
        %dst = pto.castptr %c8320_i64 : i64 -> !pto.ptr<f32, ub>
        pto.vsts %max, %dst[%i], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
      }
    }

    pto.vecscope {
      scf.for %j = %c0 to %c1 step %c1 {
        %mask, %next = pto.plt_b32 %c16_i32 : i32 -> !pto.mask, i32
        %lhs = pto.castptr %c64_i64 : i64 -> !pto.ptr<f32, ub>
        %lhs_vec = pto.vlds %lhs[%j] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
        %rhs = pto.castptr %c8320_i64 : i64 -> !pto.ptr<f32, ub>
        %rhs_vec = pto.vlds %rhs[%j] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
        %sum = pto.vadd %lhs_vec, %rhs_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
        %dst = pto.castptr %c64_i64 : i64 -> !pto.ptr<f32, ub>
        pto.vsts %sum, %dst[%j], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
      }
    }
    return
  }
}
