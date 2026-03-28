## 1. Matcher 与契约基础

- [x] 1.1 审查 `compare/select`、`tile_scalar`、`broadcast_row_binary`、`reduce_colsum` 当前 family DSL、template metadata 和 lowering descriptor，列出已丢失的语义角色 / variant 信息
- [x] 1.2 调整 `MatchRequest` 或等价 descriptor 构造逻辑，使其在模板归一化前保留 scalar 方向、full-tile vs row-broadcast 角色以及 `binary` variant 语义
- [x] 1.3 为 template 选择路径补充 family-logic 级断言或检查，防止语义角色在 matcher 阶段被静默折叠

## 2. Compare/Select Mask 契约

- [x] 2.1 为 `tcmp`、`tcmps`、`tsel`、`tsels` 明确并落地 byte-mask canonical form 的模板契约，包括 `0 == false`、`nonzero == true` 和 tail lane 约束
- [x] 2.2 调整 compare/select lowering，使其只在输入满足已批准的 byte-mask contract 时进入 OpLib path，并对不满足契约的情况给出确定性失败
- [x] 2.3 补充 `test/oplib/` 回归，覆盖 compare-to-select round-trip、非 0/1 mask byte 解释，以及非法 mask 输入的 negative case

## 3. Direction-Sensitive Family 对齐

- [x] 3.1 调整 `tile_scalar` family 的 lowering / template 选择，确保 `scalarPos` 或等价语义不会因 `vector.splat` 统一骨架而丢失
- [x] 3.2 调整 `broadcast_row_binary` family 的 matcher 与模板 contract，使其能区分“与 `dst` 同 shape 的 operand”和“row-broadcast operand”，并覆盖顺序敏感 op
- [x] 3.3 补充 `tdivs`、`trowexpanddiv`、`trowexpandsub` 等 direction-sensitive 用例的 smoke 和 negative 回归，验证反向输入不会被错误拒绝或错误归一化

## 4. Reduction Variant 语义

- [x] 4.1 评估当前 `reduce_colsum` skeleton 是否足以表达 `variant_id=binary` 的 `tmp` / staged accumulation 语义，并形成实现或收缩结论
- [x] 4.2 若可表达，则为 `tcolsum(binary)` 实现独立 variant contract；若不可表达，则同步收缩 manifest、template metadata 和 lowering，使 `binary` 不再标记为 implemented
- [x] 4.3 补充 `tcolsum(linear)` / `tcolsum(binary)` 的选择与门禁回归，确保 `binary` 不会静默回落到 `linear`

## 5. 文档与验证

- [ ] 5.1 更新 `docs/tile_fusion/a5_oplib_v1_authoring.md` 及相关文档，说明 family 逻辑 gap 的分类、byte-mask contract 和 direction-sensitive family 约束
- [ ] 5.2 运行并记录本 change 相关验证，包括 generator 自检、针对 `test/oplib` 的 family logic 回归，以及新增 negative gate 用例
