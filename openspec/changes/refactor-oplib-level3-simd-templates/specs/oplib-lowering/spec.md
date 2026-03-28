## ADDED Requirements

### Requirement: Level-3 template import must enforce the unified SIMD template contract

`PTOInstantiateAndLowerToLibCall` 导入 `oplib/level3` 模板时 MUST 对统一模板契约执行硬校验，包括 64-lane 数据向量约束、非标量相关 family 的 SIMD-only 主体数据路径约束，以及 scalar-related family 的 SIMD 化标量接入约束。

#### Scenario: Reject non-compliant non-scalar template bodies

- **WHEN** 导入的 Level-3 模板属于非标量相关 family，且模板体仍使用逐元素 `memref.load/store` 或非 64-lane 数据向量作为主体实现
- **THEN** 模板导入 MUST 失败，并向调用方返回硬错误，而不是继续接受该模板参与候选匹配

#### Scenario: Accept scalar ABI while preserving SIMD body

- **WHEN** 导入的 Level-3 模板在 ABI 上包含 scalar 参数
- **THEN** lowering/importer MUST 允许该模板参与匹配，但模板体仍 MUST 满足 64-lane SIMD 数据路径约束

### Requirement: Lowering must preserve existing match keys for concrete instances generated from shared template sources

统一模板源生成的 concrete 实例 MUST 继续沿用现有 lowering 匹配键空间，包括 `kind`、`op`、`dtype`、`variant_id`、`cmpMode`、`scalarPos`、`requiredVariantId` 和 `isBinary`。模板源组织方式的变化 MUST NOT 改变 `MatchRequest` 到 concrete 实例的选择语义。

#### Scenario: Select compare instance by existing condition key

- **WHEN** compare family 由统一 skeleton source 生成多个 condition concrete 实例
- **THEN** lowering MUST 继续通过现有 `cmpMode` 语义选择正确实例，而不是引入新的 condition 匹配协议

#### Scenario: Select dtype-specific instance from shared source expansion

- **WHEN** 同一 skeleton source 生成多个 concrete dtype 实例
- **THEN** lowering MUST 继续基于现有 `dtype` 和 `variant_id` 语义选择实例，并保持实例化缓存键对 concrete 参数类型的区分能力
