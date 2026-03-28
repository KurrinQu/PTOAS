## Why

### 概述

当前 `oplib/level3` 模板在 SIMD 约束、lane 宽度、dtype 覆盖和模板复用方式上仍然不一致，已经影响 OP-Lib 模板的可维护性、可验证性和后续扩展效率。现有模板同时存在 `vector` 路径与 `memref.load/store` 标量回退路径，且同一计算模式被拆成大量按 `op`、`dtype`、条件分散维护的文件，无法支撑后续统一演进。

### 背景与动机

仓库已经将 A5 OP-Lib vector lowering 逐步收敛到固定 64-lane SIMD 模型，但 `oplib/level3` 仍保留历史兼容模板，导致以下问题：

- 同一 family 内同时存在 32-lane 与 64-lane 模板，模板行为与后端期望不一致。
- 多个模板文件只在 `dtype` 或 compare 条件上不同，主体循环和 SIMD 骨架重复，维护成本高。
- 部分模板仍依赖 `memref.load/store` 逐元素路径，难以形成统一的 SIMD 编程范式和导入门禁。
- compare/select/bitwise family 的条件和 dtype 组合扩展缺少统一模板源，后续补类型或补条件时容易继续复制文件。

### 目标

- 将 `oplib/level3` 模板统一到 64-lane SIMD 范式。
- 以“同一计算模式一套模板源”的方式重构模板组织，覆盖 8/16/32 位宽度 dtype 与 compare 条件组合。
- 将模板约束固化为可校验的 OpenSpec 契约，避免后续继续引入按 `dtype` / 条件平铺的重复模板。
- 明确 lowering/import/instantiation 对新模板体系的兼容要求，保证后续实现有可执行边界。

### 非目标

- 本 change 不直接实现 `oplib/level3` 的代码重构。
- 本 change 不改变 PTO IR 的基础类型系统，不将 `rows/cols/v_row/v_col` 从 `i64` 改为其他位宽。
- 本 change 不引入单个 `func.func` 层面的真泛型模板机制；统一模板指单一 skeleton source，可生成多个 concrete dtype 实例。

### 预期结果

- 后续实现阶段可以基于统一 skeleton source 生成 `i8/i16/i32` 及对应支持的浮点/整数具体模板实例。
- compare family 可以用同一套模板源覆盖 `LT/LE/GT/GE/EQ/NE` 等条件，而不是继续维护按条件平铺的模板文件。
- lowering 和模板导入规则将明确哪些 family 必须使用 64-lane SIMD，哪些场景才允许保留标量输入语义。

### 成功标准

- OpenSpec 中存在明确的 `oplib-templates` 能力定义，覆盖 SIMD 范式、64-lane 规则、模板复用和 dtype/condition 覆盖要求。
- `oplib-lowering` 规格同步定义与新模板体系相匹配的导入、匹配、实例化约束。
- 后续实现任务可以直接据此拆分模板重构、验证和回归工作，不再需要补充基础需求澄清。

## What Changes

- 新增 `oplib-templates` capability，定义 `oplib/level3` 的统一 SIMD 模板体系。
- 规定 Level-3 OP-Lib 模板默认采用 64-lane SIMD 数据向量；非标量相关 family 不再使用 `memref.load/store` 作为主体计算路径。
- 规定相同计算模式的 OP 使用同一套模板源，模板源可以生成多个 concrete dtype 实例，而不是人工维护按 `dtype`/条件拆分的重复模板。
- 规定同一 compare 模板源支持 `LT/LE/GT/GE/EQ/NE` 等条件组合。
- 明确模板覆盖的 dtype 范围为 8/16/32 位宽度的各种 dtype，但仍受具体 OP 语义和后端支持矩阵约束。
- 修改 `oplib-lowering` capability，使模板导入、匹配和实例化契约能够接纳统一模板源生成的 concrete 实例，并对 SIMD 约束执行硬校验。

## Capabilities

### New Capabilities

- `oplib-templates`: 定义 `oplib/level3` 统一 SIMD 模板体系的组织方式、64-lane 约束、dtype/条件覆盖规则，以及模板源与 concrete 实例之间的契约。

### Modified Capabilities

- `oplib-lowering`: 调整 OP-Lib 模板导入、匹配和实例化要求，使其与统一 64-lane SIMD 模板体系保持一致，并对不符合模板契约的输入执行硬失败。

## Impact

- 受影响代码目录：`oplib/level3/`、`lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp`、必要时 `lib/PTO/Transforms/PTOToEmitC.cpp` 的模板兼容路径。
- 受影响测试目录：`test/oplib/`，尤其是模板导入负测、compare/select family、bitwise family、generic shape 和 emitc 相关用例。
- 受影响文档：`openspec/specs/oplib-lowering/spec.md`，并新增 `openspec/changes/refactor-oplib-level3-simd-templates/specs/oplib-templates/spec.md`。
