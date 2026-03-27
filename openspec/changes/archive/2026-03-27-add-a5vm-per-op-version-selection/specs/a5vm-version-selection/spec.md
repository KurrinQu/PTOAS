# A5VM Version Selection Specification

## ADDED Requirements

### Requirement: A5VM version selection MUST annotate every candidate PTO op before backend lowering

在 A5 fusion mainline 中，系统 MUST 在 `PTOFusionRegionGen` 之后、`PTOToA5VM` 之前运行独立的版本选择 pass。该 pass MUST 遍历当前函数中的全部 A5VM candidate PTO op，并为每个 op 写入 `pto.a5vm_lowering_choice` typed attribute。该属性 MUST 作为单一复合决策载体，同时包含 `update_mode` 与 `loop_shape` 两个字段。

#### Scenario: Selection pass annotates ops inside and outside `pto.fusion_region`

- **WHEN** 一个函数在 `PTOFusionRegionGen` 之后同时包含 `pto.fusion_region` 内部 PTO op 与 region 外 residual PTO op
- **THEN** 版本选择 pass MUST 同时遍历这两类 op
- **AND** MUST 为每个 A5VM candidate PTO op 写入 `pto.a5vm_lowering_choice`
- **AND** MUST NOT 仅在 `pto.fusion_region` 上写默认策略来替代 per-op 标注

### Requirement: `pto.a5vm_lowering_choice` MUST fully describe the selected lowering variant

`pto.a5vm_lowering_choice` MUST 采用 PTO dialect typed attribute 表达正式版本决策。该属性 MUST 同时包含以下字段：

- `update_mode`，取值 MUST 为 `post_update` 或 `no_post_update`
- `loop_shape`，取值 MUST 为 `one_d` 或 `two_d`

系统 MUST 将这两个字段视为一个不可拆分的 lowering 决策，缺失任一字段或使用未定义取值都属于非法 IR。

#### Scenario: Incomplete or invalid choice attribute is rejected

- **WHEN** 某个 A5VM candidate PTO op 携带的 `pto.a5vm_lowering_choice` 缺失字段，或字段值不属于定义好的枚举集合
- **THEN** 消费该属性的 backend pass MUST 将其视为非法 IR
- **AND** MUST NOT 以默认值、空值或独立 string attr 继续推断版本

### Requirement: `PTOToA5VM` MUST directly consume per-op selection and fail fast on missing decisions

在 A5 fusion mainline 中，`PTOToA5VM` MUST 将 `pto.a5vm_lowering_choice` 视为每个 A5VM candidate PTO op 的唯一正式版本决策输入。对已标注 op，`PTOToA5VM` MUST 直接生成属性指定的 codegen 版本；对缺失属性、属性非法或属性与 op 组合不合法的 case，`PTOToA5VM` MUST 立即失败并输出诊断。

#### Scenario: Lowering emits the selected variant without version-selection control flow

- **WHEN** `PTOToA5VM` 处理一个带有合法 `pto.a5vm_lowering_choice` 的 PTO op
- **THEN** lowering MUST 直接生成属性指定的 `post_update/no_post_update` 与 `1D/2D` 版本
- **AND** MUST NOT 再为了在这些版本之间选择而生成新的 `scf.if` / `else`

#### Scenario: Missing or incompatible choice causes lowering failure

- **WHEN** `PTOToA5VM` 遇到缺失 `pto.a5vm_lowering_choice` 的 A5VM candidate PTO op，或属性组合与该 op 的合法 lowering 版本不兼容
- **THEN** pass MUST `signalPassFailure`
- **AND** MUST 在诊断中包含 op 名称以及导致失败的属性信息

### Requirement: Fusion mainline MUST NOT use `--a5vm-lowering-strategy` as a competing source of truth

在启用 A5 fusion mainline 时，per-op `pto.a5vm_lowering_choice` MUST 是正式 lowering 决策的唯一真相源。系统 MUST NOT 再允许 `--a5vm-lowering-strategy` 覆盖、替代或补齐已进入 fusion mainline 的 per-op 决策。

#### Scenario: Fusion mainline ignores legacy global strategy

- **WHEN** 用户在启用 fusion mainline 的同时指定 `--a5vm-lowering-strategy`
- **THEN** driver MUST 明确说明该参数在 fusion mainline 中被忽略
- **AND** `PTOToA5VM` MUST 继续以 `pto.a5vm_lowering_choice` 作为正式输入
