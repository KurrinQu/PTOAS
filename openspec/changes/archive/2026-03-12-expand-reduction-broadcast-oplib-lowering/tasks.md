## 1. 范围与前置

- [x] 1.1 在 proposal/design 中固定 4.6-4.7 范围并声明依赖 `generalize-oplib-template-capabilities`

## 2. 4.6-4.7 Family 模板实现

- [x] 2.1 在 `oplib/level3/` 下补齐 `l3_reduce_row_template`(必须采用simd编程范式)
- [x] 2.2 在 `oplib/level3/` 下补齐 `l3_reduce_col_template`(必须采用simd编程范式)
- [x] 2.3 在 `oplib/level3/` 下补齐 `l3_reduce_colsum_template`(必须采用simd编程范式)
- [x] 2.4 在 `oplib/level3/` 下补齐 `l3_broadcast_row_template`(必须采用simd编程范式)
- [x] 2.5 在 `oplib/level3/` 下补齐 `l3_broadcast_col_template`(必须采用simd编程范式)
- [x] 2.6 在 `oplib/level3/` 下补齐 `l3_broadcast_row_binary_template`(必须采用simd编程范式)
- [x] 2.7 在 `oplib/level3/` 下补齐 `l3_scalar_expand_template`(必须采用simd编程范式)

## 3. Lowering 接入

- [x] 3.1 在 matcher / instantiation 中接入 4.6-4.7 family
- [x] 3.2 为 `tcolsum` 接入 `isBinary=false/true` 双 variant 选择

## 4. 测试、文档与验证

- [x] 4.1 为 12 个 op 补齐静态 valid shape 与动态 valid shape IR 测试
- [x] 4.2 增加 family 级 EmitC 终态测试，覆盖 `trowsum`、`tcolsum(true)`、`trowexpandmul`、`texpands`
- [x] 4.3 修正 `docs/PTO_IR_manual.md` 中 `trowsum`、`trowmax`、`trowmin` 的 `tmp` 文档事实
- [x] 4.4 验证 reduction / broadcast 新增 family 与既有 OP-Lib regression 一致通过
