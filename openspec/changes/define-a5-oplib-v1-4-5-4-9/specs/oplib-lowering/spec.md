## ADDED Requirements

### Requirement: A5 OpLib V1 lowering must target the explicit 4.5-4.9 operator set

当 `PTOInstantiateAndLowerToLibCall` 以 A5 OpLib V1 模式运行时，lowering 目标集 MUST 对齐 `PTO_IR_manual.md` 第 4.5~4.9 节的 in-scope op 集，并以 A5 manifest 的 `implemented` / `deferred` 状态决定行为。

#### Scenario: Lower implemented in-scope operator through generated concrete template

- **WHEN** 某个 4.5~4.9 in-scope op 在 A5 manifest 中为 `implemented`
- **THEN** lowering MUST 基于 generated concrete 模板完成候选匹配、实例创建和 call rewrite，而不是保留原 op

#### Scenario: Deferred in-scope operator fails deterministically

- **WHEN** 某个 4.5~4.9 in-scope op 在 A5 manifest 中为 `deferred`
- **THEN** lowering MUST 发出确定性诊断并失败，且 MUST NOT 将该 op 当作“暂时没有模板、先保留原 op”处理

### Requirement: Lowering must preserve existing matcher semantics for generated family templates

即使模板作者接口切换到 Family DSL，lowering 在选择 concrete 实例时 MUST 保持现有 matcher key 语义不变，包括 `kind`、`op`、`dtype`、`variant_id`、`cmpMode`、`scalarPos`、`requiredVariantId` 和 `isBinary`。

#### Scenario: Compare and select families continue to use legacy match keys

- **WHEN** lowering 为 `tcmp`、`tcmps`、`tsel` 选择 generated concrete 模板，或显式拒绝当前不对齐的 `tsels` path
- **THEN** 实例选择 MUST 继续依赖现有 compare/select 相关 matcher key，而不是引入新的匹配协议

#### Scenario: Variant-sensitive arithmetic families preserve current selection behavior

- **WHEN** lowering 为 `tdivs`、`tcolsum` 或其他带 variant / `isBinary` 选择逻辑的 in-scope op 选择模板
- **THEN** matcher MUST 继续沿用现有 `requiredVariantId` / `isBinary` 语义选择正确实例

### Requirement: Implemented in-scope operators must not silently fallback or remain after lowering

对 A5 manifest 中标记为 `implemented` 的 4.5~4.9 in-scope op，OpLib lowering SHALL 是强制性的。若 descriptor 构造、模板导入、候选选择、实例创建或 call rewrite 任一阶段失败，pass MUST 失败。

#### Scenario: Missing template candidate is a hard failure

- **WHEN** 某个 `implemented` in-scope op 在 lowering 过程中没有找到合法模板候选
- **THEN** pass MUST 发出错误并失败，而不是保留原 op 或仅输出 warning

#### Scenario: Residual in-scope operator after pass is forbidden

- **WHEN** `PTOInstantiateAndLowerToLibCall` 完成后，普通用户函数中仍残留某个 `implemented` in-scope PTO op
- **THEN** pass MUST 发出错误并失败

### Requirement: Generated concrete templates must remain the importer contract

Family DSL 和 generator 只改变模板维护方式，不改变 importer 输入边界。`PTOInstantiateAndLowerToLibCall` MUST 继续导入 concrete `.mlir` 模板，而不是直接解释 DSL 源。

#### Scenario: Op-lib directory still points to concrete templates

- **WHEN** 用户通过 `--op-lib-dir` 启用 A5 OpLib V1 lowering
- **THEN** lowering MUST 继续从该目录读取 concrete `.mlir` 模板文件，并将这些 concrete 函数作为模板注册表输入
