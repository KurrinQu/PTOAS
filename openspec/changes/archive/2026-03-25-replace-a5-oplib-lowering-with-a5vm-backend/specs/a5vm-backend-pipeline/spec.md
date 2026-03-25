# A5VM Backend Pipeline Specification

## ADDED Requirements

### Requirement: A5 backend mainline MUST lower through `PTOToA5VM`

`ptoas` 的 A5 backend 主线 MUST 以 `createLowerPTOToA5VMPass()` 作为正式 backend lowering 边界，而不是继续以旧的 OP-Lib lowering passes 作为默认 A5 backend 主线。

#### Scenario: A5VM backend path uses `PTOToA5VM` and skips legacy OP-Lib lowering passes

- **WHEN** 用户通过 `ptoas` 选择 A5VM backend 路径编译 A5 输入
- **THEN** pass pipeline MUST 在 backend lowering 边界运行 `createLowerPTOToA5VMPass()`
- **AND** MUST NOT 将 `PTOInstantiateAndLowerToLibCall` 或 `PTOInlineLibCall` 作为该 A5 backend 主线的一部分运行

### Requirement: A5 post-lowering fusion MUST run after `PTOToA5VM`

一旦 A5 backend 主线切到 `PTOToA5VM`，`PTOLowLevelLoopFusion` MUST 位于 `PTOToA5VM` 之后，并以 A5VM lowering 后的低层 loop 结构为输入契约。

#### Scenario: Low-level loop fusion consumes A5VM low-level IR

- **WHEN** A5 backend 主线在 fusion 打开时执行 post-lowering 优化
- **THEN** `PTOLowLevelLoopFusion` MUST 在 `PTOToA5VM` 之后运行
- **AND** MUST 以 `scf.for + a5vm.*` 低层结构作为正式输入
- **AND** MUST NOT 继续把旧 `pto.simd.vec_scope` / `vector.masked_*` bridge IR 当作该主线的正式输入契约

### Requirement: A5 EmitC output MUST remain available without legacy OP-Lib backend dependency

`EmitC` 作为可选输出路径 MAY 继续保留，但 A5 的 `EmitC` 路径 MUST 与旧 OP-Lib backend 主线解耦。

#### Scenario: A5 EmitC path does not require legacy OP-Lib passes

- **WHEN** 用户选择 `--pto-backend=emitc` 且目标架构为 A5
- **THEN** `ptoas` MAY 继续进入 EmitC/C++ 输出路径
- **AND** MUST NOT 要求 `PTOInstantiateAndLowerToLibCall` 或 `PTOInlineLibCall` 成为进入该输出路径的前置条件
- **AND** MUST NOT 将 `--op-lib-dir` 视为该 A5 EmitC 路径的正式 backend 必填输入
