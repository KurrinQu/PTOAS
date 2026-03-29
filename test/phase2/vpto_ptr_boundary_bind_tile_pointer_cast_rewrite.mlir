// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-text --print-ir-after-all %s -o /dev/null 2>&1 | sed -n '/IR Dump After PTOVPTOPtrBoundary/,/IR Dump After /p' | FileCheck %s

// CHECK-LABEL: IR Dump After PTOVPTOPtrBoundary
// CHECK: func.func @bind_tile_pointer_cast_boundary(%arg0: !pto.mask)
// CHECK-NOT: pto.pointer_cast
// CHECK-NOT: pto.bind_tile
// CHECK: %[[BASE0:.+]] = pto.castptr %{{.+}} : i64 -> !pto.ptr<f32, ub>
// CHECK: %[[LOAD:.+]] = pto.vlds %[[BASE0]][%{{.+}}] : !pto.ptr<f32, ub> -> !pto.vec<64xf32>
// CHECK: %[[BASE1:.+]] = pto.castptr %{{.+}} : i64 -> !pto.ptr<f32, ub>
// CHECK: pto.vsts %[[LOAD]], %[[BASE1]][%{{.+}}], %arg0 : !pto.vec<64xf32>, !pto.ptr<f32, ub>, !pto.mask
// CHECK-LABEL: func.func @bind_tile_castptr_boundary()
// CHECK-NOT: pto.pointer_cast
// CHECK-NOT: pto.bind_tile
// CHECK: %[[PTR:.+]] = pto.castptr %{{.+}} : i64 -> !pto.ptr<f32, ub>
// CHECK: pto.copy_ubuf_to_gm %[[PTR]], %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64

module {
  func.func @bind_tile_pointer_cast_boundary(%mask: !pto.mask) {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %buf = pto.pointer_cast(%c0_i64) : memref<1x64xf32, #pto.address_space<vec>>
    %tile = pto.bind_tile %buf, %c1, %c64 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x64xf32, #pto.address_space<vec>> -> memref<1x64xf32, #pto.address_space<vec>>
    %v = pto.vlds %tile[%c0] : memref<1x64xf32, #pto.address_space<vec>> -> !pto.vec<64xf32>
    pto.vsts %v, %tile[%c0], %mask : !pto.vec<64xf32>, memref<1x64xf32, #pto.address_space<vec>>, !pto.mask
    return
  }

  func.func @bind_tile_castptr_boundary() {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %buf = pto.pointer_cast(%c0_i64) : memref<1x64xf32, #pto.address_space<vec>>
    %tile = pto.bind_tile %buf, %c1, %c64 {config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>} : memref<1x64xf32, #pto.address_space<vec>> -> memref<1x64xf32, #pto.address_space<vec>>
    %src = pto.castptr %tile : memref<1x64xf32, #pto.address_space<vec>> -> !pto.ptr<f32, ub>
    %dst = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, gm>
    pto.copy_ubuf_to_gm %src, %dst, %c0_i64, %c1_i64, %c64_i64, %c0_i64, %c64_i64, %c64_i64 : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
    return
  }
}
