# a5vm-vld-vst-addressing Specification

## ADDED Requirements

### Requirement: `vld*/vst*` 无状态与 predicate 变体 MUST 接受 memref/ptr 双地址形态

A5VM 的 `vld*/vst*` 无状态与 predicate 变体在地址操作数上 MUST 同时接受 `memref` 与 `!llvm.ptr`，不得要求调用方为 memref 路径额外引入新 op 名称。

#### Scenario: Stateless/predicate load-store accepts memref operands

- **WHEN** `vlds/vldas/vldus/vsld/vldx2` 或 `vsts/vsst/vstx2/vsstb/vsta/vstas/vstar` 与 `plds/pld/pldi/psts/pst/psti` 使用 memref 地址操作数
- **THEN** IR 校验 MUST 通过
- **AND** MUST 保持既有地址空间合法性约束

#### Scenario: Existing pointer-based authoring remains valid

- **WHEN** 同一批 op 使用 `!llvm.ptr` 地址操作数
- **THEN** IR 校验 MUST 继续通过
- **AND** MUST NOT 作为 deprecated 路径被拒绝

### Requirement: A5 backend 主线 MUST 采用 memref-first 地址模型

在 `PTOToA5VM -> PTOLowLevelLoopFusion -> PTOFlattenFusionRegion` 阶段，A5 backend 主线 MUST 以 memref 作为默认地址载体，不得在优化阶段前主动退化为 pointer-only。

#### Scenario: Mainline preserves memref until emission boundary

- **WHEN** 高层 PTO 输入经 A5 backend 主线降级
- **THEN** 在上述阶段内地址模型 MUST 保留 memref 语义
- **AND** MUST NOT 为了满足发射 ABI 提前强制 pointer 化

### Requirement: 发射边界 MUST 提供等价 pointer ABI 映射

A5VM text/LLVM 发射边界 MUST 负责将 memref 地址语义映射到目标 pointer ABI，且该映射不得改变最终内建调用语义。

#### Scenario: memref and pointer forms emit equivalent intrinsic behavior

- **WHEN** 输入 IR 分别采用 memref 与 `!llvm.ptr` 两种地址形态表达等价的 `vld*/vst*` 计算
- **THEN** 发射结果 MUST 对应等价的 pointer ABI 调用语义
- **AND** MUST NOT 因地址形态不同产生行为分叉

### Requirement: Stateful `base/base_out` MUST 保持 pointer-only

`pstu/vstu/vstus/vstur` 的 `base/base_out` 在本 change 中 MUST 继续维持 pointer-only 契约，不得接受 memref。

#### Scenario: Stateful memref base/base_out is rejected

- **WHEN** `pstu/vstu/vstus/vstur` 使用 memref 作为 `base` 或 `base_out`
- **THEN** IR 校验 MUST 拒绝该输入
- **AND** 诊断 MUST 明确该限制属于 stateful base 语义边界

### Requirement: 非目标 buffer op MUST 保持既有地址契约

未纳入本 change 的 `copy_* / gather / scatter` 等 buffer op MUST 保持现有地址接口与行为，不得因本 change 被隐式放宽或收紧。

#### Scenario: Out-of-scope ops remain unchanged

- **WHEN** 编译路径包含 `copy_* / gather / scatter` 等非 `vld*/vst*` op
- **THEN** 它们的地址操作数契约 MUST 与变更前保持一致
