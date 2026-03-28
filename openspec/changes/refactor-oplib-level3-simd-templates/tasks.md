## 1. 模板导入与约束收紧

- [x] 1.1 盘点 `oplib/level3` 现有 family，标注哪些属于非标量相关 family、哪些属于 ABI 上显式带 scalar 的 family。
- [x] 1.2 在 `PTOLowerToOpLibCalls.cpp` 中收紧模板体校验，去掉非标量相关 family 对 `memref.load/store` 逐元素路径的兼容口。
- [x] 1.3 在 `PTOLowerToOpLibCalls.cpp` 和相关导入逻辑中将 64-lane 设为 Level-3 模板的唯一前向约束，并补充失败诊断。
- [x] 1.4 为统一模板源生成的 concrete 实例保留现有 `kind`、`op`、`dtype`、`variant_id`、`cmpMode`、`scalarPos`、`requiredVariantId`、`isBinary` 匹配语义。

## 2. skeleton source 与实例展开基础设施

- [x] 2.1 设计并落地 `oplib/level3` 的 skeleton source 组织方式，明确单一维护源和 concrete 实例输出位置。
- [x] 2.2 实现从 skeleton source 生成 concrete dtype/condition/variant 模板实例的脚本或生成入口。
- [x] 2.3 为 binary、tile-scalar、unary、compare 等主要计算模式定义统一 skeleton 参数维度，包括 dtype、condition、core op 和 variant。
- [x] 2.4 建立生成结果一致性检查，避免 skeleton source 与 concrete 模板实例漂移。

## 3. 分 family 迁移 `oplib/level3`

- [x] 3.1 先迁移 compare family，使 `tcmp` / `tcmps` 使用同一套 compare skeleton source 覆盖 `LT/LE/GT/GE/EQ/NE` 条件。
- [x] 3.2 迁移 int binary / int tile-scalar / int unary family，使 `i8/i16/i32` concrete 实例来自统一 skeleton source。
- [x] 3.3 迁移 float binary / float tile-scalar / float unary family，使 `f16/f32` concrete 实例来自统一 skeleton source，并统一 64-lane vector 体。
- [x] 3.4 迁移 reduction、broadcast、select 和 ternary family，去除现有 `memref.load/store` 标量 fallback，并改为 SIMD 合法实现。
- [x] 3.5 清理 `oplib/level3` 中仅因 `dtype`、condition 或核心算子不同而重复维护的旧模板文件与旧模板体。

## 4. 测试、文档与回归验证

- [x] 4.1 更新 `test/oplib` 中与模板导入、lane 校验、compare/select family、bitwise family 相关的正反向用例。
- [x] 4.2 为统一模板源覆盖的 dtype/condition 组合补充最小 smoke 测试，验证实例选择与 lowering 结果不变。
- [x] 4.3 运行至少一轮针对 `oplib/level3` 的 lit 回归，并在必要时补充 emitc/generic shape 相关回归。
- [x] 4.4 更新 `docs/tile_fusion/` 或其他相关文档，说明 Level-3 模板已采用“单一 skeleton source + concrete 实例展开”的组织方式和 64-lane 约束。
