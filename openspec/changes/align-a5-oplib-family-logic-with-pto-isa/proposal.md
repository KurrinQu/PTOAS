## Why

### 概述

当前 `oplib/level3` 与同级 `pto-isa` 在 family 级 OP 实现逻辑上的对齐仍存在三处高优先级真问题：`compare/select` 的 mask 表示、`broadcast_row_binary` / `tile_scalar` 的 family 形态收敛，以及 `tcolsum(binary)` 为代表的 reduction 执行路径差异。

### 背景与动机

上一轮对齐工作主要收敛了 dtype 支持范围，但 review 继续表明，`implemented` 并不自动等价于“family 逻辑已经与 A5 对齐”。

现阶段至少还有三类风险：

1. `compare/select` family 在 OpLib 中采用 byte-per-element mask 规范形态，而 A5 `TCMP/TCMPS/TSEL/TSELS` 使用 packed predicate 与 `psts/plds/vsel` 路径；如果不把这种差异写成明确契约，后续很容易把表示层差异误当成语义等价，或者反过来把合法 canonicalization 误判成 drift。
2. `tile_scalar` 与 `broadcast_row_binary` family 当前为了统一模板骨架，固定采用 `vector.splat` 和角色收敛，但 PTO-ISA 公共 API 实际允许更宽的参数形态与部分操作数顺序切换，典型如 `TDIVS` 的双入口、`TROWEXPANDDIV` / `TROWEXPANDSUB` 的方向敏感实现；如果 matcher 和 spec 不把这些边界讲清楚，OpLib 会继续在“可接受 family 形态”上比 A5 更窄。
3. reduction family 尤其 `tcolsum(binary)`，OpLib 当前模板表达为统一的逐行累加骨架，而 A5 原生实现包含 `tmp`、binary tree 和 `mem_bar` 协调；如果继续只对齐“结果语义”而不约束 variant 语义，就会让 `isBinary` 这类 matcher 轴逐步失去工程含义。

如果现在不把这三类问题整理成单独 change，后续实现很容易继续把 family 逻辑差异混入 dtype、manifest 或单点 bug 修复，导致规范边界越来越模糊。

## What Changes

### 目标

- 为 A5 OpLib V1 补一层 family-logic 对齐规范，明确哪些差异属于合法 canonical form，哪些属于必须补齐的实现 / matcher gap。
- 为 `compare/select` family 建立明确契约：定义 OpLib byte-mask 规范形态与 A5 packed predicate 语义之间的可接受映射关系，以及何时必须保留 packed-predicate 级别的行为约束。
- 为 `tile_scalar`、`broadcast_row_binary` 等 family 建立参数角色与运算方向约束，避免当前模板形态继续无意收窄 PTO-ISA 已接受的 family 输入空间。
- 为 reduction family 尤其 `tcolsum(binary)` 建立 variant 级语义要求，明确 `linear` / `binary` 何时可以共用 skeleton，何时必须体现 A5 `tmp` / binary-tree / barrier 语义。
- 补充文档、spec 和回归门禁，使后续实现能明确判断某个 gap 属于“表示层差异可接受”、“family 边界需扩展”，还是“variant 语义需要真对齐”。

### 非目标

- 不在本 change 中直接补齐所有模板实现或 lowering 代码。
- 不重新设计 Level-3 `Family DSL + snippet + skeleton` 作者模型。
- 不扩大到 A5 之外架构，也不处理 4.5~4.9 范围外 PTO op。
- 不把所有 A5 intrinsic 执行细节逐条原样内嵌回 OpLib 模板；本 change 只定义“哪些必须对齐，哪些允许 canonicalize”。

### 预期结果

- PTOAS 内部对 family 逻辑 gap 有统一分类，不再把 `vec_scope` 内的所有差异一概归类成“只是写法不同”。
- `compare/select`、broadcast/tile-scalar、reduction 三组高优先级 family 能形成明确 spec 契约，指导后续模板、matcher、lowering 和回归收敛。
- 后续实现可以基于 spec 直接判断：某个 family 是否允许保持 MLIR canonical form，还是需要提升为更接近 A5 的实现 / variant 行为。

## Capabilities

### New Capabilities

- `oplib-family-logic-alignment`: 约束 A5 OpLib V1 在 family 级实现逻辑上与同级 `pto-isa` 的可接受对齐边界，覆盖 mask 表示、参数角色/操作数方向，以及 reduction variant 语义。

### Modified Capabilities

- `oplib-lowering`: 调整 family 级 matcher、variant 选择和强制门禁，确保 lowering 不会把当前未对齐的 family 形态误判为已完全等价。

## Impact

### 预期影响

- 受影响源码主要包括 `oplib/level3/skeletons/`、`oplib/level3/families/`、`lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp`、`lib/PTO/IR/PTO.cpp`、`test/oplib/` 与相关文档。
- 受影响 family 重点集中在 `compare/select`、`tile_scalar`、`broadcast_row_binary`、`reduce_colsum`，并可能波及相邻 family 的 matcher 轴和 template metadata。
- 受影响回归包括模板选择 smoke、family negative case、`tcolsum(binary)` 变体选择，以及 compare/select mask round-trip 行为验证。

### 成功标准

- 新 change 能把三类真问题拆成可执行的 spec / design / tasks，而不是停留在 review 结论。
- spec 明确区分“表示层允许差异”和“family / variant 级必须补齐”的边界。
- 后续实现完成后，模板、matcher、lowering 与回归可以稳定回答这三个问题：mask 如何表示、family 角色是否被错误收窄、`binary` variant 是否真的保留 A5 语义。
