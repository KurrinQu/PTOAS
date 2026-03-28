## 1. A5 V1 范围与真值层

- [x] 1.1 盘点 `PTO_IR_manual.md` 第 4.5~4.9 节的 op 标题集合，形成 V1 in-scope operator 列表。
- [x] 1.2 设计并落地 `pto-isa` A5 自动对齐 manifest schema，至少包含 op、family、A5 状态、dtype、关键约束和语义来源路径。
- [x] 1.3 实现从 `pto-isa` 提取 4.5~4.9 A5 约束的同步脚本，并生成 checked-in manifest snapshot。
- [x] 1.4 为 manifest 中的 `implemented` / `deferred` 状态定义明确的校验规则和失败诊断。

## 2. Family DSL 与生成器基础设施

- [x] 2.1 设计并落地声明式 Family DSL，覆盖 family、参数角色、dtype 轴、variant 轴、metadata 和 matcher key。
- [x] 2.2 定义 Mixed-Body MLIR snippet 合同，分别覆盖 binary、tile-scalar、unary、ternary、compare、select、reduction、broadcast family。
- [x] 2.3 重构 `oplib/level3/generate_level3_templates.py` 或等价入口，使其从 Family DSL + snippet 生成 concrete `.mlir` 模板。
- [x] 2.4 保留 `--write` / `--check` 工作流，并补齐漂移检测。

## 3. 编译器消费侧改造

- [x] 3.1 更新 `PTOLowerToOpLibCalls`，使其继续消费 concrete 模板，同时支持新的 family 划分和 manifest 驱动的 in-scope / deferred 语义。
- [x] 3.2 更新 `PTOValidateSimdIR`，为新生成模型补齐 family 级约束和失败诊断。
- [x] 3.3 更新 `PTOToEmitC`，补足 4.5~4.9 in-scope family 所需的 A5 vector EmitC 覆盖。
- [x] 3.4 确保 `kind/op/dtype/variant_id/cmpMode/scalarPos/requiredVariantId/isBinary` 的实例选择语义保持兼容。

## 4. 分 family 迁移首批 op

- [x] 4.1 迁移 4.5 binary / tile-scalar / unary family，包括 float 与 int 路径。
- [x] 4.2 迁移 4.5 ternary、partial binary 和激活相关 family，包括 `tprelu`、`tlrelu`。
- [x] 4.3 迁移 4.6 reduction family：`trowsum/trowmax/trowmin/tcolsum/tcolmax/tcolmin`。
- [x] 4.4 迁移 4.7 broadcast family：`trowexpand/tcolexpand/trowexpandmul/trowexpanddiv/trowexpandsub/texpands`。
- [x] 4.5 迁移 4.8 compare/select family 的 active lowering：`tcmp/tcmps/tsel`，并保留 `tsels` 的模板资产与后续对齐入口。
- [x] 4.6 迁移 4.9 bitwise family：`tand/tor/txor/tshl/tshr/tnot`。

## 5. 测试、文档与验证

- [x] 5.1 为每个 4.5~4.9 family 补齐至少一条 positive lit 回归。
- [x] 5.2 为 Family DSL、manifest、template import、lowering 失败路径补齐 negative 回归。
- [x] 5.3 增加 `implemented` op 必有 concrete 模板与 lowering 用例的对齐测试。
- [x] 5.4 运行 `openspec validate`、模板生成检查和至少一轮 `test/oplib` / `test/tile_fusion` 相关 lit 回归。
- [x] 5.5 更新 `docs/tile_fusion/` 或等价文档，说明 A5 OpLib V1 的作者接口、manifest 对齐和 4.5~4.9 范围边界。
