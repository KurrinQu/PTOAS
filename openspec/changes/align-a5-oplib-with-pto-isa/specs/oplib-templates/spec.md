## ADDED Requirements

### Requirement: A5 OpLib concrete templates SHALL match the validated `pto-isa` semantic matrix

`oplib/level3` concrete templates MUST覆盖同级 `pto-isa` A5 已确认支持的 dtype / variant 组合，不得长期只提供单一 `f32` 子集而把其余已实现语义留在模板之外。

#### Scenario: Compare and select families expand beyond `f32`

- **WHEN** `tcmp`、`tcmps`、`tsel`、`tsels` 在同级 `pto-isa` A5 语义中支持 `i32/u32/f32/i16/u16/f16/i8/u8`
- **THEN** `oplib/level3` 对应 family MUST 提供与这些 dtype 对齐的 concrete template 覆盖，而不是仅保留 `f32`

#### Scenario: Reduction, broadcast, and scalar-expand families align with A5 dtype support

- **WHEN** `trowsum`、`trowmax`、`trowmin`、`tcolsum`、`tcolmax`、`tcolmin`、`trowexpand`、`tcolexpand`、`trowexpandmul`、`trowexpanddiv`、`trowexpandsub`、`texpands` 在同级 `pto-isa` A5 头文件中声明了特定 dtype 支持范围
- **THEN** `oplib/level3` 的 family template MUST 以同一范围生成 concrete 模板，至少不得把已支持 dtype 静默收缩为 `f32-only`

### Requirement: Template metadata SHALL NOT overclaim unsupported A5 dtypes

Concrete template 的 `pto.oplib.match.dtype` 和 manifest 投影 MUST 只声明同级 `pto-isa` A5 真正支持的 dtype，不得因为快照错误把不合法 dtype 暴露为 implemented 覆盖。

#### Scenario: Bitwise templates stay integer-only

- **WHEN** `tand`、`tor`、`txor`、`tnot` 的同级 `pto-isa` A5 语义只允许整数宽度类型
- **THEN** `oplib/level3` template 和关联 manifest metadata MUST 只声明整数 dtype，不得把 `f16`、`f32` 或 `bf16` 记为 supported

#### Scenario: Family-specific dtype restrictions are preserved

- **WHEN** 某个 family 在同级 `pto-isa` A5 语义中只支持子集 dtype，例如 `tlrelu` 仅支持 `f16/f32`，`trsqrt` 仅支持 `f16/f32`，`trowsum` 仅支持 `f16/f32`
- **THEN** concrete template MUST 按该子集生成，不得扩张到未被 A5 语义允许的 dtype

### Requirement: Decomposition-backed template coverage SHALL be explicit

对于没有独立 A5 `_IMPL`、但存在公共 API 语义重写或 PTOAS 明确采纳 decomposition 语义的 OP，模板接入 MUST 显式记录来源和边界，不能以“看起来能拼出来”作为 implemented 依据。

#### Scenario: Public API semantic rewrite can justify template coverage

- **WHEN** 某个 OP 在同级 `pto-isa` 公共 API 中已被稳定映射到已有 A5 语义，例如 `trecip` 映射到 `TDIVS(dst, 1, src)`
- **THEN** `oplib/level3` MAY 为其提供等价 template 覆盖，但 MUST 在 manifest 和回归中明确该覆盖来源于公共 API 语义，而不是独立 A5 `_IMPL`

#### Scenario: Missing A5 `_IMPL` without approved decomposition stays out of implemented templates

- **WHEN** 某个 OP 仅有公共 API 入口，但缺少 A5 `_IMPL`，且当前 change 未批准其 decomposition 语义进入 OpLib，例如 `taddc`、`tsubc`、`taddsc`、`tsubsc`
- **THEN** `oplib/level3` MUST NOT 将其直接纳入 implemented concrete template 集
