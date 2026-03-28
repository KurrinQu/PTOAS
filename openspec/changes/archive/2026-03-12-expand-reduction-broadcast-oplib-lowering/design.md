# Design: 4.5-4.6 Reduction 与 Broadcast OP-Lib Lowering

## 范围与依赖

本设计覆盖：

1. 4.5 Reduction Operations
2. 4.6 Broadcast Operations

依赖 `generalize-oplib-template-capabilities` 已支持：

1. `(src, tmp, dst)` 与 `(scalar, dst)` 模板签名
2. vector reduction 相关模板体与 EmitC 支持
3. family-specific attr 匹配

## Family 划分

### Family A: `l3_reduce_row_template`

覆盖：

- `trowsum`
- `trowmax`
- `trowmin`

策略：

1. 统一使用 `(src, tmp, dst)` 签名。
2. variant-only。
3. 输出形状固定为列向量语义。

### Family B: `l3_reduce_col_template`

覆盖：

- `tcolmax`
- `tcolmin`

策略：

1. 使用 `(src, dst)` 签名。
2. variant-only。
3. 输出形状固定为行向量语义。

### Family C: `l3_reduce_colsum_template`

覆盖：

- `tcolsum`

策略：

1. 使用 `(src, tmp, dst)` 签名。
2. variant 由 `pto.oplib.match.is_binary` 区分 `false/true`。

### Family D: `l3_broadcast_row_template`

覆盖：

- `trowexpand`

策略：

1. `(src, dst)` 签名。
2. src 语义上是列向量。

### Family E: `l3_broadcast_col_template`

覆盖：

- `tcolexpand`

策略：

1. `(src, dst)` 签名。
2. src 语义上是行向量。

### Family F: `l3_broadcast_row_binary_template`

覆盖：

- `trowexpandmul`
- `trowexpanddiv`
- `trowexpandsub`

策略：

1. 统一 `(src0, src1, dst)` 签名。
2. `src1` 语义上是每行一个标量的列向量。
3. 允许 seed 覆盖这 3 个 op。

### Family G: `l3_scalar_expand_template`

覆盖：

- `texpands`

策略：

1. 使用 `(scalar, dst)` 签名。
2. variant-only。

## 文档收敛

manual 需要与代码事实对齐：

1. `trowsum` 需要 `tmp`
2. `trowmax` 需要 `tmp`
3. `trowmin` 已声明 `tmp`，保持不变

## 测试

必须覆盖：

1. 全部 12 个 op 的静态 valid shape 与动态 valid shape 命中
2. reduction output shape 正确
3. `tmp` 参与路径
4. `tcolsum(isBinary=false/true)` 双 variant
5. family 级 EmitC 终态检查：`trowsum`、`tcolsum(true)`、`trowexpandmul`、`texpands`
