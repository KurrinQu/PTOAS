# OP-Lib 开发者最小指南：编写 Mixed IR 模板（`vector/arith/pto.simd`）

- 状态：Draft v1.1
- 适用范围：Level-3 OP-Lib Binary Element-Wise（`tadd/tsub/tmul/tdiv/tmax/tmin`）
- 目标读者：编写/维护 OP-Lib 模板的开发者

当前 A5 OpLib V1 的完整作者入口已经升级为 `Family DSL + snippet + manifest`。
请先阅读
[`docs/tile_fusion/a5_oplib_v1_authoring.md`](./a5_oplib_v1_authoring.md)；
本文保留为 mixed IR / SIMD 模板体约束的最小补充指南，不再代表完整 V1 范围与对齐规则。

维护说明：

1. 当前 Level-3 模板主维护源位于 `oplib/level3/skeletons/`。
2. importer-active concrete 模板位于 `oplib/level3/*.mlir`，由 `oplib/level3/generate_level3_templates.py` 统一展开生成。
3. 新建或重构的 Level-3 模板统一使用 64-lane SIMD 向量体；即使 ABI 上保留 scalar 参数，也必须通过 `vector.splat` 等 SIMD 手段并入计算。

## 1. 先记住四条硬约束

1. 对外 ABI 不变：函数签名必须是 `(!pto.tile_buf, !pto.tile_buf, !pto.tile_buf) -> ()`。
2. 模板函数体必须非空，禁止空 body 回退。
3. V1.1 只支持 `f16/f32 + row_major`。
4. 函数体只能使用 allowlist IR：`arith/vector/memref(受限，不含load/store)/scf`、`pto.simd.tile_to_memref`、`pto.simd`（可选）。

## 2. 你现在有两种合法写法

1. 纯 vector 写法：直接用 `vector + arith + scf + memref`。
2. `pto.simd` 写法：在需要显式 mask/predicate/post-update 语义时使用 `pto.simd.*`。

说明：两种写法都属于 OP-Lib 官方可用 IR，不要求统一降级到 `vector` 后再继续。

## 3. 最小上手流程

1. 先决定模板角色：`variant` 或 `seed`。
2. 写齐 `pto.oplib.*` 匹配元数据。
3. 选择模板体写法：纯 vector 或 `pto.simd`。
4. 若使用 `pto.simd.predicate/load/store/load_pu/store_pu`，再补 `pto.simd.level/lanes`。
5. 对 `seed`，只保留一个 `pto.simd.core_slot = "binary_ewise_core"` 的核心算术 op。
6. 用 `ptoas` 跑一轮导入和实例化自检；只有模板不在默认 installed/repo `oplib/level3` 时才显式加 `--op-lib-dir=...`。

## 4. 角色差异（`variant` vs `seed`）

| 维度                   | `variant`                                                       | `seed`                                                                                                                                 |
| ---------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `pto.oplib.entry_role` | `"variant"`                                                     | `"seed"`                                                                                                                               |
| 需要的操作元数据       | `pto.oplib.op`, `pto.oplib.variant_id`, `pto.oplib.match.dtype` | `pto.oplib.seed_id`, `pto.oplib.seed_dtype`, `pto.oplib.seed.support_dtypes`, `pto.oplib.seed.support_ops`, `pto.oplib.seed.core_slot` |
| 核心算术 op 的改写     | 不改写，直接使用模板体算术语义                                  | 实例化时仅改写 core slot 对应算术 op                                                                                                   |
| 推荐使用场景           | 某个 op 的专门优化实现                                          | 一套骨架覆盖多个 binary op                                                                                                             |

## 5. 最小模板骨架 A：纯 vector 写法（推荐先跑通）

```mlir
func.func private @__pto_oplib_seed_vec_bin_core(
    %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %dst:  !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    attributes {
      pto.oplib.kind = "l3_binary_elementwise_template",
      pto.oplib.entry_role = "seed",
      pto.oplib.seed_id = "seed_vec_bin_core",
      pto.oplib.seed_dtype = "f32",
      pto.oplib.seed.support_dtypes = ["f16", "f32"],
      pto.oplib.seed.support_ops = ["tadd", "tsub", "tmul", "tdiv", "tmax", "tmin"],
      pto.oplib.seed.core_slot = "binary_ewise_core",
      pto.oplib.match.rows = -1 : i64,
      pto.oplib.match.cols = -1 : i64,
      pto.oplib.match.blayout = "row_major",
      pto.oplib.match.slayout = "any",
      pto.oplib.match.fractal = -1 : i64,
      pto.oplib.cost = 10 : i64,
      pto.oplib.priority = 0 : i64
    } {
  %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>
  %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>
  %md = pto.simd.tile_to_memref %dst  : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>

  %flat0 = memref.reinterpret_cast %m0 to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
  %flat1 = memref.reinterpret_cast %m1 to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
  %flatd = memref.reinterpret_cast %md to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>

  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c64 = arith.constant 64 : index
  scf.for %i = %c0 to %c1024 step %c64 {
    %a = vector.load %flat0[%i] : memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, vector<64xf32>
    %b = vector.load %flat1[%i] : memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, vector<64xf32>
    %c = arith.addf %a, %b {pto.simd.core_slot = "binary_ewise_core"} : vector<64xf32>
    vector.store %c, %flatd[%i] : memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, vector<64xf32>
  }
  return
}
```

## 6. 最小模板骨架 B：`pto.simd` 写法（需要显式 mask 时）

```mlir
// 在骨架 A 基础上，把 vector.load/store 替换为：
//   pto.simd.predicate + pto.simd.load/store
// 并在函数属性中增加：
//   pto.simd.level = "binary_ewise_v1"
//   pto.simd.lanes = 64 : i64
```

补充：若只需要告诉 CodeGen 在循环外层放置 `__VEC_SCOPE__`，可使用：

```mlir
pto.simd.vec_scope {
  scf.for %i = %c0 to %c1024 step %c64 {
    // vector.load/store + arith.*f
  }
}
```

## 7. 示例模板约束检查表（CR/自测直接打勾）

| #   | 检查项                  | 必须规则                                                                                | 常见失败码                                             |
| --- | ----------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| 1   | 外部 ABI                | 输入 3 个 `!pto.tile_buf`，返回 `()`                                                    | `E_OPLIB_EMPTY_BODY_FOR_SIMD`（常与空体/导入失败伴随） |
| 2   | 模板体非空              | 函数非 external 且 body 非空                                                            | `E_OPLIB_EMPTY_BODY_FOR_SIMD`                          |
| 3   | dtype 范围              | 仅 `f16/f32`                                                                            | `E_OPLIB_SIMD_UNSUPPORTED_DTYPE`                       |
| 4   | layout 范围             | 仅 `row_major`                                                                          | `E_OPLIB_SIMD_UNSUPPORTED_LAYOUT`                      |
| 5   | IR allowlist            | 仅使用约定可用 IR；`memref.load/store` 虽允许导入，但建议仅在确有必要时使用             | `E_OPLIB_BODY_DISALLOWED_IR`                           |
| 6   | 核心槽位唯一性          | 恰好一个 `pto.simd.core_slot = "binary_ewise_core"`                                     | `E_OPLIB_SIMD_INVALID_CORE_SLOT`                       |
| 7   | 核心 op 类型            | 必须是 `arith.addf/subf/mulf/divf/maximumf/minimumf` 之一                               | `E_OPLIB_SIMD_INVALID_CORE_SLOT`                       |
| 8   | `pto.simd` 属性（按需） | 使用 `pto.simd.predicate/load/store/load_pu/store_pu` 时，必须有 `pto.simd.level/lanes` | `E_OPLIB_SIMD_ATTR_REQUIRED`                           |
| 9   | lanes 一致性（按需）    | `pto.simd` 相关向量宽度与 `pto.simd.lanes` 一致                                         | `E_OPLIB_SIMD_LANES_MISMATCH`                          |
| 10  | seed 改写约束           | seed 实例化时只能改写 core slot 对应算术                                                | `E_OPLIB_SIMD_INVALID_CORE_SLOT`                       |
| 11  | 实例函数必须有 body     | 禁止 fake-body fallback                                                                 | `E_OPLIB_INSTANCE_BODY_MISSING`                        |

## 8. 最小自检命令

1. 导入与选择阶段检查（看是否能产生实例）：

```bash
ptoas test/tile_fusion/softmax_chain.pto \
  --op-lib-dir=<your-oplib-dir> \
  --dump-ir-after-oplib-lowering -o -
```

2. 融合后检查（确认模板体 IR 仍符合预期；不再要求必须 `simd->vector`）：

```bash
ptoas test/tile_fusion/softmax_chain.pto \
  --op-lib-dir=<your-oplib-dir> \
  --dump-ir-after-op-fusion -o -
```

3. 快速定位常见失败：

```bash
ptoas <input.pto> --op-lib-dir=<your-oplib-dir> -o /tmp/out.cpp 2>&1 | \
  rg "E_OPLIB_|no matching OP-Lib entry|error:"
```

## 9. 高频踩坑

1. `pto.oplib.match.blayout` 写成 `row_major`，但模板匹配目标 tile 不是 `row_major`。
2. seed 里打了多个 `pto.simd.core_slot`，实例化改写失败。
3. 使用了 allowlist 之外的 dialect/op，导入阶段直接失败。
4. 使用 `pto.simd.predicate/load/store/load_pu/store_pu` 但漏写 `pto.simd.level/lanes`。
5. 模板体继续使用 `builtin.unrealized_conversion_cast`（V1.2 起不再接受）。
6. 空模板体还期望走历史 fallback（V1.1 已明确禁止）。
