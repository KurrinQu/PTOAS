// RUN: { ptoas %S/../oplib/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all --print-ir-after-all-func-filter=flash_attention_softmax_block -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After {{.*}}PTOViewToMemrefPass
// CHECK-LABEL: func.func @flash_attention_softmax_block
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK-DAG: pto.tmuls ins(%{{[0-9]+}}, %{{[A-Za-z0-9_]+}} : memref<{{.*}}>, f32) outs(%{{[0-9]+}} : memref<{{.*}}>)
// CHECK-DAG: pto.tadds ins(%{{[0-9]+}}, %{{[A-Za-z0-9_]+}} : memref<{{.*}}>, f32) outs(%{{[0-9]+}} : memref<{{.*}}>)
// CHECK-DAG: pto.tdivs ins(%{{[0-9]+}}, %{{[A-Za-z0-9_]+}} : memref<{{.*}}>, f32) outs(%{{[0-9]+}} : memref<{{.*}}>)
// CHECK-NOT: !pto.tile_buf
// CHECK-NOT: pto.alloc_tile
// CHECK: return
// CHECK: IR Dump After PTOCreateFusionGroups
