# Level-3 OP-Lib IR 接口规范（V2.0：Multi-Family）

- 状态：Draft v2.0
- 生效范围：PTOAS Level-3 A5 OP-Lib lowering
- 目标读者：PTOAS OP-Lib 维护者、模板开发者、tile fusion pass 维护者

当前 A5 OpLib V1 的作者接口、manifest 对齐规则和 4.5~4.9 范围边界，优先参考
[`docs/tile_fusion/a5_oplib_v1_authoring.md`](./a5_oplib_v1_authoring.md)。
本文保留为 Level-3 multi-family importer / metadata 契约说明。

## 1. 目标与范围

本文定义 Level-3 OP-Lib 的多 family 模板接口规范。

目标如下：

1. 保持 OP-Lib 对外 ABI 为 `!pto.tile_buf` 主导，不为每个 family 单独发明新的调用协议。
2. 将模板导入、匹配、实例化、inline 从 binary-only 模型推广到多 family、可变参数模型。
3. 明确 `pto.oplib.kind`、签名矩阵、`argN.*` 匹配元数据和 family-specific attr matching 的契约。
4. 将模板源码统一收敛到 `oplib/level3/`，并采用“单一 skeleton source + concrete 实例展开”的维护方式。
5. 约束 A5 OP-Lib vector EmitC 的最小支持集合，只放开 4.4-4.8 当前需要的能力。

本文不定义具体某个业务 op 的 skeleton；具体 op family 划分见各 change 的 proposal/design。

## 2. 总体契约

### 2.1 统一入口键

所有模板继续使用 `pto.oplib.kind` 作为主分派键，不引入第二层总开关。

legacy family：

1. `l3_binary_elementwise_template`

新增 family 示例：

1. `l3_float_binary_elementwise_template`
2. `l3_float_tile_scalar_template`
3. `l3_float_unary_template`
4. `l3_reduce_row_template`
5. `l3_cmp_tile_tile_template`
6. `l3_int_binary_elementwise_template`

### 2.2 统一 ABI 约束

模板函数、实例函数、调用点都要求：

1. 返回类型固定为 `()`
2. 参数个数由 `kind` 决定
3. tile-like 参数使用 `!pto.tile_buf<...>`
4. scalar-like 参数使用 builtin scalar type，例如 `f32`、`i32`、`index`

### 2.3 Level-3 skeleton source 组织方式

`oplib/level3` 采用“单一维护源 + concrete 输出”的目录约定：

1. skeleton source 放在 `oplib/level3/skeletons/`
2. importer-active concrete 模板文件放在 `oplib/level3/*.mlir`
3. 统一生成入口为 `oplib/level3/generate_level3_templates.py`

约束如下：

1. 同一计算模式的公共骨架只在 `skeletons/` 维护一份。
2. `dtype`、compare condition、core op、variant 等差异通过生成展开到 concrete 实例。
3. lowering/importer 只依赖 concrete `func.func` 与既有匹配键，不直接消费 skeleton source。

## 3. `kind` 与签名矩阵

Level-3 多 family 首版支持以下固定签名类别：

1. `(tile, tile, dst)`
2. `(tile, scalar, dst)`
3. `(tile, tile, tile, dst)`
4. `(tile, scalar, tile, dst)`
5. `(tile, dst)`
6. `(scalar, dst)`
7. `(src, tmp, dst)`
8. `(mask, src0, src1, dst)`
9. `(src0, src1, selectMode, dst)`

说明：

1. `dst`、`mask`、`tmp` 在 ABI 层都仍是 `!pto.tile_buf`，差异体现在 family 语义和 metadata。
2. `selectMode`、`scalar` 使用 builtin scalar type。
3. legacy `l3_binary_elementwise_template` 继续沿用 `(tile, tile, dst)`。

模板导入时按 `kind` 驱动签名校验，不再使用固定 “3 个 `tile_buf` 参数” 的全局规则。

## 4. 元数据约定

### 4.1 通用元数据

所有模板继续使用：

1. `pto.oplib.kind`
2. `pto.oplib.entry_role`
3. `pto.oplib.op`
4. `pto.oplib.variant_id`
5. `pto.oplib.match.dtype`
6. `pto.oplib.cost`
7. `pto.oplib.priority`
8. `pto.oplib.sync`
9. `pto.oplib.seed.*`

### 4.2 legacy 匹配元数据

仅 `l3_binary_elementwise_template` 继续兼容：

1. `pto.oplib.match.rows`
2. `pto.oplib.match.cols`
3. `pto.oplib.match.blayout`
4. `pto.oplib.match.slayout`
5. `pto.oplib.match.fractal`

### 4.3 `argN.*` 匹配元数据

除 legacy binary family 外，其余新 family 一律使用按参数编号的匹配元数据：

1. `pto.oplib.match.argN.rows`
2. `pto.oplib.match.argN.cols`
3. `pto.oplib.match.argN.blayout`
4. `pto.oplib.match.argN.slayout`
5. `pto.oplib.match.argN.fractal`

约束如下：

1. `N` 按函数参数编号，从 `0` 开始。
2. 仅 tile-like 参数允许声明 `argN.*`。
3. 若 `kind` 的第 `N` 个参数是 scalar，模板上声明 `argN.*` 视为硬错误。
4. 新 family 的 tile-like 参数必须完整声明该组元数据；缺失任一项都视为硬错误。
5. `rows/cols/fractal` 允许使用 `-1` 表示 wildcard。
6. `blayout/slayout` 允许的字符串值为 `row_major`、`col_major`、`none_box`、`any`。

### 4.4 family-specific attr matching

按需使用以下属性型匹配元数据：

1. `pto.oplib.match.scalar_pos`
2. `pto.oplib.match.cmp_mode`
3. `pto.oplib.match.is_binary`

约束如下：

1. `scalar_pos` 必须指向一个 scalar 参数位置，否则为硬错误。
2. `l3_cmp_tile_tile_template` 与 `l3_cmp_tile_scalar_template` 必须提供 `cmp_mode`。
3. `l3_reduce_colsum_template` 必须提供 `is_binary`。
4. 其他 family 是否使用上述属性，由该 family 的 design 定义；未声明需求时可以省略。

## 5. Seed 与实例化规则

### 5.1 variant / seed

继续保留两类入口：

1. `variant`
2. `seed`

`variant` 要求：

1. 提供 `pto.oplib.op`
2. 提供 `pto.oplib.variant_id`
3. 提供 `pto.oplib.match.dtype`

`seed` 要求：

1. 提供 `pto.oplib.seed_id`
2. 提供 `pto.oplib.seed_dtype`
3. 提供 `pto.oplib.seed.support_dtypes`
4. 提供 `pto.oplib.seed.support_ops`
5. 可选提供 `pto.oplib.seed.core_slot`

### 5.2 core slot 约束

首版 seed 仍服务于单一 core-slot family。

当前强约束：

1. 使用 `pto.simd.core_slot` 的模板必须且仅能有一个 core slot op。
2. 现阶段 seed core slot 仍限定为 float binary arithmetic：
   1. `arith.addf`
   2. `arith.subf`
   3. `arith.mulf`
   4. `arith.divf`
   5. `arith.maximumf`
   6. `arith.minimumf`
3. 非单 core-slot family 不应复用首版 seed 改写模型。

### 5.3 实例 key

实例函数缓存键由以下部分组成：

1. `variant_id`
2. family-specific attr choices
3. 所有 concrete argument types

因此实例化不再假设固定 3 个参数。

## 6. 模板体可用 IR 集合

### 6.1 允许的 dialect / op

模板体首版允许：

1. `arith.*`
2. `vector.*`
3. `memref.*`
4. `scf.*`
5. `pto.simd.tile_to_memref`
6. `pto.simd.vec_scope`
7. `pto.simd.predicate/load/store/load_pu/store_pu`
8. `pto.simd.reduction`
9. `math.exp`
10. `math.log`
11. `math.sqrt`
12. `math.rsqrt`

### 6.2 明确不允许

以下情况会在模板导入阶段硬失败：

1. 空模板体
2. `builtin.unrealized_conversion_cast`
3. `memref.load`
4. `memref.store`
5. 未列入白名单的 `math.*`，例如 `math.sin`
6. 不在允许集合中的其他 dialect / op

补充约束：

1. 非标量相关 family 不允许以 `memref.load/store` 作为逐元素 fallback。
2. ABI 上显式带 scalar 的 family 允许保留 scalar 参数，但模板体仍必须通过 `vector.splat` 或等价 SIMD 手段并入计算。

## 7. `pto.simd` 与 A5 vector 规则

### 7.1 `pto.simd` 属性

Level-3 新模板统一使用 64-lane SIMD 约束：

1. 数据向量必须是 `vector<64xT>`
2. mask 向量必须与 64-lane 数据向量对应
3. 不再为新模板新增 32-lane 前向能力

模板体若使用 `pto.simd.predicate/load/store/load_pu/store_pu`，必须满足：

1. `pto.simd.level`
2. `pto.simd.lanes`

若模板使用 core slot，还需要：

1. `pto.simd.core_slot`

### 7.2 A5 vector 路径属性

对于 A5 OP-Lib vector lowering，模板开发者需要显式提供：

1. `vector.load` / `vector.maskedload`
   1. `pto.simd.vld_dist`
2. `vector.store` / `vector.maskedstore`
   1. `pto.simd.vst_dist`
   2. 值必须以 `DIST_` 开头
3. vector float unary / binary arithmetic
   1. `pto.simd.exec_mode`
   2. 值必须以 `MODE_` 开头

### 7.3 A5 最小支持矩阵

当前 A5 OP-Lib vector EmitC 最小支持集合：

1. float vector load/store
2. float vector unary
3. float vector binary arithmetic
4. `math.exp/log/sqrt/rsqrt`
5. vector compare
6. vector select
7. `pto.simd.reduction`（直接映射 A5 reduce 指令，结果保持向量形态）
8. `vector.reduction`（保留兼容路径，必要时经 helper 转标量）
9. integer vector load/store legality检查
10. integer vector bitwise / shift lowering

动态 shape 仅覆盖动态 valid shape，不扩展到动态 physical tile shape。

## 8. 校验规则

模板导入阶段会执行以下硬校验：

1. 签名与 `kind` 对应的签名矩阵一致
2. 新 family 的 tile-like 参数完整声明 `argN.*`
3. `scalar_pos/cmp_mode/is_binary` 在对应 family 上存在且合法
4. 模板体不出现白名单外 IR
5. 使用 `pto.simd.*` 时 `lanes` 与相关 vector 类型一致
6. A5 vector 类型、element type、lane 组合必须在允许集合内

典型错误码：

1. `E_OPLIB_EMPTY_BODY_FOR_SIMD`
2. `E_OPLIB_SIMD_LANES_MISMATCH`
3. `E_OPLIB_SIMD_INVALID_CORE_SLOT`
4. `E_OPLIB_SIMD_UNSUPPORTED_DTYPE`
5. `E_OPLIB_SIMD_UNSUPPORTED_LAYOUT`
6. `E_OPLIB_INSTANCE_BODY_MISSING`
7. `E_OPLIB_BODY_DISALLOWED_IR`
8. `E_OPLIB_SIMD_ATTR_REQUIRED`

## 9. 目录与测试约束

Level-3 模板目录统一为：

1. `oplib/level3/`
2. `oplib/level3/skeletons/` 作为 skeleton source 主维护目录
   lit 约束：

3. `ptoas` 默认应解析到 installed/repo `oplib/level3/`，显式 `--op-lib-dir` 仅用于覆盖
4. `test/tile_fusion/oplib/` 不再维护第二份模板源
5. 基础设施负测资源放在 `test/tile_fusion/resources/`

编译性能约束：

1. `PTOInstantiateAndLowerToLibCall` 在实例化后应移除 module 内未被直接调用的 imported/instantiated OP-Lib 私有函数，避免无关模板继续参与后续优化与 EmitC。

## 10. 开发建议

建议开发顺序：

1. 先确定 family 的 `kind` 与签名类别
2. 再补 `argN.*` 与 attr matching 元数据
3. 使用最小 Mixed Body IR skeleton 构造模板
4. 先通过导入校验，再补具体 op family 测试
