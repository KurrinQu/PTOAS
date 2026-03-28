## Why

### 概述

当前 `oplib/level3` 与同级 `pto-isa` A5 语义源之间已经出现了多处可确认的对齐偏差，导致 PTOAS 内部的 manifest、模板覆盖和实际 A5 语义能力不再一致。

### 背景与动机

现有问题主要集中在三类：

1. `a5_oplib_v1_manifest.yaml` 与同级 `pto-isa` 当前实现不一致，例如 `trecip` 已有公共 API 和 A5 ST 用例，但仍被标记为 `deferred`。
2. 多个已实现 OP 的 OpLib concrete 模板只覆盖了 `pto-isa` A5 真正支持矩阵的子集，典型集中在 compare/select、reduction、broadcast、scalar-expand，以及部分 arithmetic/bitwise family。
3. 对 `taddc`、`tsubc`、`taddsc`、`tsubsc` 这类上游缺少 A5 `_IMPL` 的 OP，目前缺少明确约定：哪些必须继续 `deferred`，哪些允许按 decomposition 语义对齐到 OpLib / EmitC 路径。

如果继续以当前 manifest 作为“真值层”，后续新增模板、修复 lowering 或补回归时会持续放大误判和维护成本。

## What Changes

### 目标

- 重新定义 PTOAS 内 A5 OpLib 与同级 `pto-isa` 的对齐规则，明确 manifest、template、lowering 三层的职责边界。
- 修正 `a5_oplib_v1_manifest.yaml` 中已确认失真的条目，避免把 `pto-isa` 已支持 OP 误标为 `deferred`，也避免把不合法 dtype 误记为支持。
- 为 `oplib/level3` 建立可验证的模板覆盖要求，使 compare/select、reduction、broadcast、scalar-expand、bitwise 和 arithmetic family 的 dtype/variant 覆盖与 A5 真实语义来源一致。
- 明确 ternary 与 `trecip` 一类 OP 的对齐策略：区分“必须等待 A5 原生 `_IMPL`”和“允许通过 decomposition 对齐到现有语义”的边界。
- 补充对应的回归和对齐检查，确保后续不再出现 manifest、模板和上游语义漂移。

### 非目标

- 不扩展 A5 之外的架构范围。
- 不引入新的 Level-1 / Level-2 作者 DSL。
- 不在本 change 中处理 4.5~4.9 范围外的 PTO op。
- 不要求一次性补齐所有 `pto-isa` 已有但当前 PTOAS 未消费的额外扩展 op。

### 预期结果

- PTOAS 能基于同级 `pto-isa` 的当前 A5 语义，稳定回答“某个 OP 是否 implemented、支持哪些 dtype/variant、为何 deferred”。
- `oplib/level3` 的 concrete 模板覆盖范围与 A5 真实语义矩阵保持一致，至少不再因为 manifest 失真或模板缺位造成静默收缩。
- 对需要 decomposition 的 OP，PTOAS 内部有一致的规范和测试，而不是靠零散的 EmitC 特例维持行为。

## Capabilities

### New Capabilities

- `oplib-templates`: 约束 `oplib/level3` concrete 模板的 family、dtype、variant 覆盖必须与同级 `pto-isa` A5 语义矩阵对齐，并明确允许的 decomposition template 范围。

### Modified Capabilities

- `oplib-lowering`: 调整 A5 manifest 对齐、`implemented/deferred` 语义、以及 lowering 对 manifest/模板覆盖不一致场景的门禁要求。

## Impact

### 预期影响

- 受影响代码主要包括 `oplib/level3/`、`lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp`、`lib/PTO/IR/PTO.cpp`、`lib/PTO/Transforms/PTOToEmitC.cpp` 和 `test/oplib/`。
- 受影响数据源包括 `oplib/level3/families/a5_oplib_v1_manifest.yaml` 及其生成/校验逻辑。
- 受影响回归包括 implemented-op 对齐检查、family positive smoke、以及 compare/select、reduction、broadcast、scalar-expand、ternary/recip 的专项测试。

### 成功标准

- 新 spec 能明确区分 manifest 真值、template 覆盖和 lowering 行为的契约。
- 至少覆盖本次 review 已确认的高优先级偏差：`trecip`、compare/select dtype 覆盖、row/col reduction、broadcast、scalar-expand、bitwise manifest dtype 失真。
- change 完成后，后续实现可以基于 spec 直接判断某个 gap 是 manifest 问题、模板问题，还是上游 `pto-isa` 本身缺少 A5 `_IMPL`。
