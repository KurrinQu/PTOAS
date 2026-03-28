// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm --emit-a5vm %s -o - 2>/dev/null | FileCheck %s --check-prefix=A5VM
// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SUMMARY --check-prefix=A5VM
// RUN: ./build/tools/ptoas/ptoas --pto-arch=a5 --pto-backend=emitc %s -o - 2>/dev/null | FileCheck %s --check-prefix=EMITC

// SUMMARY: A5VM IR op: a5vm.copy_gm_to_ubuf
// SUMMARY: A5VM IR op: a5vm.copy_ubuf_to_gm
// A5VM-LABEL: func.func @pto_backend_a5vm_wiring
// A5VM: a5vm.copy_gm_to_ubuf
// A5VM: a5vm.vlds
// A5VM: a5vm.vabs
// A5VM: a5vm.vsts
// A5VM: a5vm.copy_ubuf_to_gm
// A5VM-NOT: emitc.call_opaque

// EMITC: #include "pto/pto-inst.hpp"
// EMITC-LABEL: __global__ AICORE void pto_backend_a5vm_wiring(
// EMITC: TLOAD(
// EMITC: TABS(
// EMITC: TSTORE(

module {
  func.func @pto_backend_a5vm_wiring(%src: !pto.ptr<f32>, %dst: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %src_tv = pto.make_tensor_view %src, shape = [%c32, %c32], strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>
    %dst_tv = pto.make_tensor_view %dst, shape = [%c32, %c32], strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>
    %src_slice = pto.partition_view %src_tv, offsets = [%c0, %c0], sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %dst_slice = pto.partition_view %dst_tv, offsets = [%c0, %c0], sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %tmp = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %out = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%src_slice : !pto.partition_tensor_view<32x32xf32>)
      outs(%tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tabs ins(%tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      outs(%out : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%out : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      outs(%dst_slice : !pto.partition_tensor_view<32x32xf32>)
    return
  }
}
