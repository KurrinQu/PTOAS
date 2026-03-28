## Why

### 概述

当前 PTOAS 已经具备 PTO dialect、Level-3 OP-Lib lowering、`pto.simd.*` 和 A5 vector EmitC 的基础设施，但 A5 OP-Lib 模板的作者接口仍停留在 `catalog.json + 文本 placeholder skeleton + concrete 展开文件` 的维护模式。这个模式已经能工作，但对 4.5~4.9 节这批高频 vector op 来说，维护成本、约束表达能力和与 `pto-isa` A5 语义的一致性都不够稳定。

本 change 定义 A5 OpLib V1 的 OpenSpec 契约：范围只覆盖 `PTO_IR_manual.md` 第 4.5~4.9 节的算子；实现主战场放在 `PTOAS`；作者接口升级为声明式 Family DSL；Lowering 和模板合法性以 `pto-isa` A5 自动对齐结果为真值来源。

### 背景与动机

现有 Level-3 模板体系存在以下问题：

- 模板作者需要同时维护 `catalog`、文本 skeleton 和大量 concrete `.mlir` 输出，真实维护源和编译器消费源之间存在重复和漂移风险。
- 同一计算模式的大量 `dtype`、`opname`、condition 变体仍通过生成脚本做字符串级展开，结构化约束不足，扩展新 family 时容易继续堆积 ad-hoc 逻辑。
- PTOAS 已有 `PTOValidateSimdIR`、`PTOLowerToOpLibCalls`、`PTOToEmitC`，但还没有一套以 `pto-isa` A5 为语义真源的自动对齐机制，导致“IR manual 定义了哪些 op”“A5 真正支持哪些 dtype/layout”“哪些 op 需要 deferred”缺少统一入口。
- 第一批真正需要高频维护和做高层优化的算子并不是全 PTO 集合，而是 IR manual 4.5~4.9 中的 vector arithmetic、reduction、broadcast、compare/select、bitwise 这几类。V1 如果范围不收紧，文档和实现都很容易失控。

### 目标

- 定义 A5 OpLib V1 的明确范围，只覆盖 `PTO_IR_manual.md` 第 4.5~4.9 节的 op 集。
- 将 Level-3 模板作者接口升级为声明式 Family DSL + Mixed-Body MLIR snippet，减少作者直接维护 concrete 样板代码的负担。
- 保持 `--op-lib-dir` 和 concrete `.mlir` importer 契约稳定，不在 V1 重写整个 OP-Lib 导入模型。
- 为 in-scope op 建立 `pto-isa` A5 自动对齐机制，使模板生成、lowering 选择和测试覆盖都以统一 manifest 为真值。
- 让 `PTOLowerToOpLibCalls`、`PTOValidateSimdIR` 和 `PTOToEmitC` 可以围绕这批 op 稳定演进，而不是继续堆叠零散特例。

### 非目标

- 本 change 不在 `pto-isa` 中引入 MLIR/LLVM 基建，也不把 `pto-isa` 变成 IR 库实现仓。
- 本 change 不定义新的 Level-1/Level-2 作者接口；V1 作者入口仍是 Level-3。
- 本 change 不把 PTO 全部算子纳入首批范围；4.1~4.4 以及 4.10 之后的算子不属于 V1。
- 本 change 不承诺把 `PTO_IR_manual` 中每个 4.5~4.9 算子都立即做成 A5 可 lowering 实现；若 `pto-isa` A5 语义源不足，可显式标记 `deferred`。
- 本 change 不默认新增新的 `pto.simd` public op；只有现有 `pto.simd + vector/arith/memref/scf/math` 无法表达某个 in-scope op 时才允许补最小扩展。

### 预期结果

- PTOAS 内会存在一套新的 A5 OpLib V1 文档契约，清楚定义作者接口、in-scope op 集和 `pto-isa` 自动对齐边界。
- 后续实现阶段可以把 4.5~4.9 节的 op 按 family 迁移到声明式 DSL，而不需要继续扩张字符串模板体系。
- Lowering 和验证可以对 in-scope op 做显式硬失败，不再依赖“有模板就降，没有模板就留原 op”的模糊行为。
- compare/select、reduction/broadcast、integer bitwise 等 family 可以在统一框架下演进，而不是各自维护独立规则。

### 成功标准

- OpenSpec 中存在新的 `oplib-templates` capability，明确 Family DSL、concrete 生成、in-scope op 集和 `pto-isa` A5 manifest 对齐规则。
- OpenSpec 中对 `oplib-lowering` 的修改明确规定 4.5~4.9 in-scope op 的 lowering、deferred 和失败行为。
- 新 change 的 `proposal/design/tasks/specs` 足够完整，后续实现不需要再回到“V1 覆盖哪些 op / 放在哪个仓 / 作者写哪一层 / 与 `pto-isa` 怎么对齐”这些基础问题上重新拍板。

## What Changes

- 新增 `oplib-templates` capability，定义 A5 OpLib V1 的作者模型、模板生成模型和 4.5~4.9 首批覆盖范围。
- 修改 `oplib-lowering` capability，定义 in-scope op 的 mandatory lowering、deferred 语义、实例选择语义和 residual-op 失败约束。
- 规定实现主战场在 `PTOAS`，而不是 `pto-isa`。
- 规定作者入口是 Level-3 Family DSL + snippet，不是继续直接维护 `catalog + placeholder skeleton`。
- 规定模板 legality 和覆盖范围以 `pto-isa` A5 自动提取/同步得到的 manifest snapshot 为真值来源。

## Capabilities

### New Capabilities

- `oplib-templates`: 定义 A5 OpLib V1 的 Family DSL、snippet 合同、generated concrete 模板、in-scope op 集，以及与 `pto-isa` A5 manifest 的自动对齐规则。

### Modified Capabilities

- `oplib-lowering`: 收紧 4.5~4.9 in-scope op 的 lowering 契约，要求基于 generated concrete 模板做实例选择，并对 deferred / missing / residual-op 给出确定性行为。

## Impact

- 受影响目录：
  - `oplib/level3/`
  - `lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp`
  - `lib/PTO/Transforms/PTOValidateSimdIR.cpp`
  - `lib/PTO/Transforms/PTOToEmitC.cpp`
  - 新增 A5 manifest 同步/检查脚本所在目录
- 受影响测试：
  - `test/oplib/`
  - 与 compare/select、reduction/broadcast、bitwise family、generic shape、emitc 相关的 lit 回归
- 受影响文档：
  - 当前 change 的 OpenSpec 产物
  - 后续实现落地后需要补 `docs/tile_fusion/` 和生成器使用说明
