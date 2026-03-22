// RUN: { ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null 2>&1 || true; } | FileCheck %s

// Driver-sample 5.4 scheduling regression for paged_attention online_update:
// after OpScheduling, the two main planned hotspots in the else-branch must
// each appear as one contiguous run span.

// CHECK-LABEL: IR Dump After OpScheduling
// CHECK-LABEL: func.func @kernel_online_update(

// 1x16 diamond hotspot: contiguous group-0 span.
// CHECK: pto.tmax{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 0 : i64
// CHECK-NEXT: pto.tsub{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 1 : i64
// CHECK-NEXT: pto.texp{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 2 : i64
// CHECK-NEXT: pto.tsub{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 3 : i64
// CHECK-NEXT: pto.texp{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 4 : i64
// CHECK-NEXT: pto.tmul{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 5 : i64
// CHECK-NEXT: pto.tmul{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 6 : i64
// CHECK-NEXT: pto.tadd{{.*}}pto.fusion.group_id = 0 : i64{{.*}}pto.fusion.order = 7 : i64

// 16x128 join hotspot: both row reshapes are pulled ahead of the group so that
// the group-1 compute members become one contiguous span.
// CHECK: %[[ROW0:[0-9]+]] = pto.treshape %{{[0-9]+}} : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16, v_row=1, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1, v_row=16, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>
// CHECK-NEXT: %[[ROW1:[0-9]+]] = pto.treshape %{{[0-9]+}} : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16, v_row=1, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1, v_row=16, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>
// CHECK-NEXT: pto.trowexpandmul ins(%{{[0-9]+}}, %[[ROW0]]{{.*}}outs(%[[JOIN0:[0-9]+]]
// CHECK-SAME: pto.fusion.group_id = 1 : i64
// CHECK-SAME: pto.fusion.order = 0 : i64
// CHECK-NEXT: pto.trowexpandmul ins(%{{[0-9]+}}, %[[ROW1]]{{.*}}outs(%[[JOIN1:[0-9]+]]
// CHECK-SAME: pto.fusion.group_id = 1 : i64
// CHECK-SAME: pto.fusion.order = 1 : i64
// CHECK-NEXT: pto.tadd ins(%[[JOIN0]], %[[JOIN1]]
// CHECK-SAME: pto.fusion.group_id = 1 : i64
// CHECK-SAME: pto.fusion.order = 2 : i64
