## 1. 基础设施模型改造

- [x] 1.1 在 `PTOLowerToOpLibCalls.cpp` 中将模板注册结构升级为 family-aware 模型
- [x] 1.2 引入 family-specific 模板签名校验，支持可变参数 ABI
- [x] 1.3 新增并解析 `pto.oplib.match.argN.*` 元数据
- [x] 1.4 新增并解析 `pto.oplib.match.scalar_pos`、`pto.oplib.match.cmp_mode`、`pto.oplib.match.is_binary`
- [x] 1.5 将 matcher / instance key / call rewrite 从 binary-only 逻辑推广到可变参数模型
- [x] 1.6 将 `PTOInlineLibCallPass` 的实例函数处理同步推广到可变参数模型

## 2. 模板与 EmitC 能力扩展

- [x] 2.1 扩大模板体 IR 白名单，仅新增 `math.exp/log/sqrt/rsqrt`
- [x] 2.2 扩大 A5 OP-Lib vector EmitC 支持，补齐 float unary 与 math unary
- [x] 2.3 扩大 A5 OP-Lib vector EmitC 支持，补齐 vector compare/select 与 vector reduction
- [x] 2.4 扩大 A5 OP-Lib vector EmitC 支持，补齐 integer vector load/store legality 与 bitwise/shift lowering
- [x] 2.5 将 lit 使用的模板源统一到 `oplib/level3/`

## 3. 测试与文档

- [x] 3.1 增加基础设施负测，覆盖 family 签名、`argN.*` 元数据与 attr matching 非法场景
- [x] 3.2 增加模板体 IR 白名单负测，覆盖未放行的 `math`、compare、reduction 与 integer vector op
- [x] 3.3 增加 A5 vector legality 与 codegen 负测，覆盖非法 dtype、lanes 或不支持的向量形式
- [x] 3.4 更新 `docs/tile_fusion/oplib_ir_spec.md` 为多 family 规范

## 4. 验证

- [x] 4.1 运行 OP-Lib 相关 lit 回归，确认现有 binary 路径在多 family 基础设施下不回归
