## ADDED Requirements

### Requirement: Level-3 OP-Lib templates must use a unified 64-lane SIMD model

`oplib/level3` 模板体系 MUST 以 64-lane SIMD 作为统一数据向量模型。除 ABI 上显式带 scalar operand 的模板类别外，模板体 MUST 以 SIMD/vector 路径完成主体计算，不得依赖 `memref.load/store` 逐元素回退。

#### Scenario: Non-scalar template uses unified SIMD body

- **WHEN** Level-3 模板属于非标量相关 family，例如 binary、unary、reduction、broadcast、compare 或 select family
- **THEN** 模板体 MUST 使用 `vector.*` / `pto.simd.*` 等 SIMD 形式完成主体数据通路，并且数据向量 MUST 为 64 lanes

#### Scenario: Scalar-related template still remains SIMD-oriented

- **WHEN** Level-3 模板在 ABI 上包含 builtin scalar 参数，例如 tile-scalar 或 select-scalar family
- **THEN** 模板体 MAY 接收 scalar 输入，但 MUST 通过 `vector.splat` 或等价 SIMD 手段并入 64-lane 计算，而不是回退为逐元素 `memref.load/store`

### Requirement: Same compute pattern must be maintained as one template source

相同计算模式的 OP MUST 共享同一套模板源。模板源 SHALL 表达公共 skeleton，并生成按 `dtype`、condition 或 variant 区分的 concrete 实例；系统 MUST NOT 要求为仅在 `dtype`、compare 条件或核心算子上不同的实例手工维护重复模板体。

#### Scenario: One binary elementwise skeleton covers multiple dtypes

- **WHEN** 一组 OP 共享同一 binary elementwise 计算模式，且这些 OP 支持 8/16/32 位宽度 dtype
- **THEN** 模板体系 MUST 使用同一 binary skeleton source 生成对应的 concrete dtype 实例，而不是为每个 dtype 手工维护一份等价模板逻辑

#### Scenario: One compare skeleton covers multiple conditions

- **WHEN** compare family 需要覆盖 `LT`、`LE`、`GT`、`GE`、`EQ`、`NE` 等条件
- **THEN** 模板体系 MUST 使用同一 compare skeleton source 生成对应条件实例，并保持每个 concrete 实例仍可被 lowering 通过既有 condition 键选择

### Requirement: Template coverage must follow supported dtype matrix without changing dimension width semantics

统一模板源 MUST 覆盖 8/16/32 位宽度的各种 dtype，但仅限于对应 OP 语义与后端支持矩阵允许的组合。`rows/cols/v_row/v_col` 在模板类型与匹配语义中 MUST 保持 `i64` 口径，不得因模板重构改为其他位宽表示。

#### Scenario: Unsupported dtype is excluded by capability boundary

- **WHEN** 某个 OP 在 PTO 语义或 backend 支持矩阵中不支持某个 8/16/32 位宽度 dtype
- **THEN** 模板体系 MAY 不生成该 dtype 的 concrete 实例，但 MUST 不影响其他已支持 dtype 继续复用同一 skeleton source

#### Scenario: Dimension semantics remain i64

- **WHEN** Level-3 模板声明 tile shape、valid shape 或匹配元数据中的 `rows/cols/v_row/v_col`
- **THEN** 这些维度语义 MUST 继续使用 `i64` 口径，与现有 `TileBufType` 和 OP-Lib 匹配实现保持一致
