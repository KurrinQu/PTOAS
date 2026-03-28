## Context

### 范围

本 design 只覆盖上一轮 review 已确认的 3 个 family 级真问题：

1. `compare/select` 的 mask 表示与 A5 packed predicate 语义之间的边界不清。
2. `tile_scalar`、`broadcast_row_binary` 等 direction-sensitive family 在模板骨架归一化后，存在收窄 PTO-ISA 可接受参数形态的风险。
3. `reduce_colsum` 尤其 `variant_id=binary`，当前模板表达与 A5 `tmp + binary tree + mem_bar` 路径不再一一对应。

不在本 design 范围内的内容：

- dtype 覆盖矩阵本身；这一部分已由前一轮 change 单独处理。
- 4.5~4.9 范围外 PTO op。
- A5 之外架构的 OpLib family 设计。

### 当前状态

当前 `oplib/level3` 的作者模型已经固定为 `Family DSL + snippet + skeleton`，generator 统一生成 `tile_to_memref`、`pto.simd.vec_scope`、tail mask 和 64-lane skeleton。这个设计使得大量 A5 helper 内部细节在模板层被 canonicalize 成统一 MLIR 形态。

这类 canonicalization 带来了两个结果：

1. 很多差异只是表示层差异，而不是语义 drift，例如 `vector.maskedload/store` 对应 A5 `CreatePredicate + vlds/vsts`。
2. 但也有少数差异已经越过“表示层”边界，开始影响 family 可接受形态或 variant 语义，例如：
   - `compare/select` 把 packed predicate 改写成 byte mask tile；
   - `broadcast_row_binary` 把角色固定为 `src0=full tile`、`src1=row-broadcast tile`；
   - `reduce_colsum(binary)` 复用 generic accumulation skeleton，而没有保留 `binary` 的执行编排含义。

### 约束

- 不废弃当前 `Family DSL + snippet + skeleton` 路线，也不回退到逐个 concrete 模板手写。
- 不把 A5 intrinsic 级 API 原样嵌回模板层；模板仍允许使用 MLIR canonical form。
- `PTOLowerToOpLibCalls` 仍通过 descriptor / matcher / template registry 选择模板，不引入新的独立 lowering 流水线。
- 只要一个 variant 继续标记为 implemented，就必须对它的 family 语义边界给出可验证定义，不能继续保留“名字上有 variant，行为上无差异”的状态。

## Goals / Non-Goals

**Goals:**

- 把 family 逻辑 gap 明确分成三类：允许的表示层 canonicalization、必须保留的参数角色语义、必须恢复的 variant 语义。
- 为 `compare/select` 补齐 byte-mask canonical form 的显式契约，避免其成为未定义行为。
- 为 `tile_scalar`、`broadcast_row_binary` 等 family 明确“语义角色记录”与“模板内部归一化”的分层。
- 为 `tcolsum(binary)` 恢复有工程含义的 variant 契约，至少不再让 `binary` 仅是 matcher 标签。
- 为后续实现提供可执行的任务拆分：模板、matcher、lowering、文档、回归各自需要做什么。

**Non-Goals:**

- 不要求 OpLib 模板逐句复刻 A5 helper 的所有 intrinsic 顺序。
- 不在本 change 中一并重做所有 unary/partial family 的 native-vs-decomposition 选择。
- 不引入新的公开 PTO IR op 或新的外部 CLI 接口。

## Decisions

### 决策 1：family 逻辑差异按“表示层 / family 形态 / variant 语义”三层分类处理

本 change 不再把所有差异统称为“与 A5 不同”，而是采用分层处理：

1. 表示层差异：
   允许在模板中保留 MLIR canonical form，只要对外语义与 A5 等价，且契约可验证。
2. family 形态差异：
   若当前模板 / matcher 收窄了 PTO-ISA 已接受的参数角色、操作数顺序或广播形态，则必须补齐 matcher 或 descriptor 语义。
3. variant 语义差异：
   若模板暴露了多个 variant，但实现上并不能区分其执行语义，则必须恢复差异或收缩实现声明。

采用该分类的原因：

- 能把 `compare/select` 和 `tcolsum(binary)` 这类问题从“都叫 gap”里拆出来，避免采取同一种修复手段。
- 能保留当前 OpLib canonical skeleton 的收益，同时阻止它继续吞掉本应保留的 family/variant 语义。

备选方案：

- 全部按 A5 helper 原样重写模板：语义最直观，但会破坏当前 generator 与 skeleton 复用路径，成本过高。
- 全部继续视为 canonicalization：最省事，但无法解释 `broadcast_row_binary` 与 `binary` variant 已经收窄的事实。

### 决策 2：`compare/select` 保留 byte-mask canonical form，但必须建立与 packed predicate 的闭合语义契约

本 change 不要求把 `compare/select` 模板改回 `psts/plds` 形态，而是正式承认：

- OpLib 内部模板可以用 byte-per-element mask 作为 canonical form。
- 但该 canonical form 必须有明确约束：`0 == false`、`nonzero == true`、tail lane 的无效元素不得泄漏为有效选择结果。
- `tcmp/tcmps` 产出的 mask 与 `tsel/tsels` 消费的 mask 必须形成可验证 round-trip，而不是“恰好当前实现能连起来”。

采用该策略的原因：

- 它保留了当前模板层对 packed predicate 细节去耦的收益。
- 它避免为模板层引入 A5 predicate pack/unpack 专有 IR 需求。
- 它把“表示层差异”关进明确边界内，而不是让后续实现自行猜测。

备选方案：

- 强制模板改回 packed predicate：更贴近 A5，但与当前模板 allowlist 和 generator 路线不一致。
- 不补契约，仅保留现状：会继续让 compare/select 处于“语义上看似对齐、规范上没有定义”的状态。

### 决策 3：direction-sensitive family 必须先记录语义角色，再进行模板归一化

对 `tile_scalar`、`broadcast_row_binary` 以及与之相邻的 direction-sensitive family，本 change 采用两阶段模型：

1. lowering descriptor / matcher 先记录原始语义角色，例如哪个 operand 与 `dst` 同 shape、哪个 operand 是 scalar / row-broadcast、当前 op 是否对操作数顺序敏感。
2. 只有在语义角色已确定后，模板内部才允许做 `vector.splat`、operand swap 或 canonical skeleton 归一化。

这意味着：

- `tile_scalar` 不能只在模板中看到一个 `%scalarVec`，还必须在 matcher 层保留 `scalarPos` 或等价语义。
- `broadcast_row_binary` 不能继续把 `src0=full tile`、`src1=row-broadcast tile` 当作唯一合法外部形态；如果 A5 允许对称输入，lowering 必须能在 family 入口处表达出来，再映射到具体模板方向。
- `tdivs`、`trowexpanddiv`、`trowexpandsub` 这类顺序敏感 op，不能靠“模板里随便 swap 一下”隐式兜底，而要有显式 variant / matcher 语义。

采用该策略的原因：

- 它允许继续复用当前 skeleton，而不必为每种角色排列都复制模板体。
- 它把“模板归一化”和“外部语义接受范围”从同一层拆开，避免 canonical form 反过来限制 API 语义。

备选方案：

- 为每种角色排列都生成独立 family：直观，但模板数量和 matcher 复杂度会快速膨胀。
- 继续只靠少量 `requiredVariantId` 特判：对 `tdivs` 勉强够用，但无法系统覆盖 row-broadcast family。

### 决策 4：`reduce_colsum(binary)` 必须恢复独立 variant 语义；若做不到，就收缩 implemented 声明

`reduce_colsum` 当前暴露 `linear` / `binary` 两个 variant，但模板 skeleton 基本等价。这个状态不能继续保留。

本 change 的要求是：

- 若 `variant_id=binary` 继续 implemented，则必须提供独立 template contract，至少显式体现 `tmp` 参与和 staged accumulation 语义，不能再与 `linear` 共用无差别 skeleton。
- 若当前模板 IR 无法在本 change 范围内表达足够清楚的 `binary` 语义，则 manifest / matcher / lowering 必须收缩 `binary`，只保留 `linear` implemented。

采用该策略的原因：

- 它恢复了 `isBinary` / `variant_id=binary` 的工程含义。
- 它避免继续以“variant 名字存在”为由误导 lowering 与回归。
- 它允许实现阶段根据成本做取舍，但不允许继续维持语义失配的 implemented 状态。

备选方案：

- 保留当前实现并在文档中弱化 `binary` 含义：最小改动，但会让 variant 轴继续失真。
- 强制在本 change 中完整复刻 A5 `mem_bar` 行为：目标明确，但实现风险较高；因此本 design 允许“收缩 implemented”作为保底策略。

### 决策 5：测试策略按“contract test + lowering smoke + negative gate”三层组织

为了让这 3 个真问题后续不再回流，本 change 的测试策略分三层：

1. contract test：
   验证新 spec 中定义的 family 语义边界，例如 byte mask round-trip、direction-sensitive operand normalization、binary variant contract。
2. lowering smoke：
   验证具体 PTO op 能命中正确 family / variant，不会因 canonical form 丢失语义角色。
3. negative gate：
   验证不合法 family 形态或未实现 variant 会硬失败，而不是静默落到错误模板。

采用该策略的原因：

- 仅靠 lit smoke 很难发现 family 语义被悄悄收窄。
- 仅靠模板检查又无法覆盖 lowering 入口的角色丢失问题。

## Risks / Trade-offs

- [Risk] `compare/select` 保留 byte-mask canonical form 会让部分维护者误以为 PTOAS 已不需要考虑 packed predicate 语义。
  → Mitigation：在 spec 和回归里显式写出 byte-mask 与 packed predicate 的映射契约，禁止把当前实现当作无约束的普通 `i8` tile 处理。

- [Risk] direction-sensitive family 的 matcher 信息增加后，`MatchRequest` 与模板选择逻辑会更复杂。
  → Mitigation：优先复用现有 `scalarPos`、`requiredVariantId`、`isBinary` 等字段；只有在无法表达时才增加新的 family-local 元数据。

- [Risk] `reduce_colsum(binary)` 若需要独立 skeleton，可能触发模板数量和验证成本上升。
  → Mitigation：允许在本 change 中以“收缩 implemented”作为保底，不强求在一次实现里完成最复杂的 variant 对齐。

- [Risk] 这一轮 change 与上一轮 dtype change 边界相近，后续实施时容易混改单。
  → Mitigation：在 tasks 中明确限制只处理 family 逻辑、matcher 语义和 variant 对齐，不重复扩展 dtype 矩阵。

## Migration Plan

1. 先补 proposal / spec / design，固定 family 逻辑边界和术语。
2. 再调整模板与 lowering 入口的语义记录方式，优先覆盖 `compare/select` 与 direction-sensitive family。
3. 对 `reduce_colsum(binary)` 做实现路径评估：
   - 能提供独立 variant contract，则继续 implemented；
   - 不能提供，则同步收缩 manifest / matcher / lowering。
4. 最后补齐 contract test、negative gate 和文档说明。

本 change 不涉及对外 CLI 或用户数据迁移。若实施阶段发现某个 variant 无法在当前框架下安全对齐，回退策略优先选择“撤回 implemented 声明”，而不是保留名义存在但语义不清的模板。

## Open Questions

- `broadcast_row_binary` 最终是通过扩展现有 matcher 字段表达角色方向，还是引入新的 family-local metadata 更稳妥？
- `reduce_colsum(binary)` 在当前 allowlist IR 下是否足以表达所需 staged accumulation，还是应在这一轮直接收缩到 `linear-only`？
- `compare/select` 的 contract test 是否需要显式覆盖“非 0 但非 1”的 mask byte，避免后续实现把 byte-mask 误收紧为“只接受 0/1”？
