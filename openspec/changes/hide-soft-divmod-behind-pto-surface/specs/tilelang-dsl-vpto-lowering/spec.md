## ADDED Requirements

### Requirement: `pto.vdiv` MUST use dtype-directed lowering

`pto.vdiv` 在 TileLang DSL lowering 中 MUST 按元素类型分流，而不是对所有 dtype 使用同一条后端路径。  
其中：

- `f16/f32` `pto.vdiv` MUST 保留 authoring-form VPTO `pto.vdiv` 路径
- `i8/i16/i32` `pto.vdiv` MUST 改写为 internal soft helper path

整数 `pto.vdiv` 的 soft helper path MAY 以 helper `func.func` + `func.call` 的形式存在于 `mlir_text()` 阶段，但该 helper call MUST 通过现有 backend-inline 主线在后续阶段被消除。

#### Scenario: floating `pto.vdiv` stays on the VPTO path

- **WHEN** 用户以 `f16` 或 `f32` 向量调用 `pto.vdiv`
- **THEN** lowering MUST 保留合法的 authoring-form `pto.vdiv`
- **AND** 后续 VPTO backend/emitter MUST 继续沿用现有 `vdiv` 指令路径

#### Scenario: integer `pto.vdiv` is rewritten to an internal soft helper

- **WHEN** 用户以 `i8/i16/i32` 向量调用 `pto.vdiv`
- **THEN** semantic/lowering MUST NOT 继续把该调用保留为 authoring-form `pto.vdiv`
- **AND** 该调用 MUST 被改写到 internal soft helper path
- **AND** 最终用户 contract MUST 仍表现为“使用 `pto.vdiv` 获得整数 vector division”

### Requirement: `pto.vmod` MUST lower through the internal soft-helper path

在本 change 中，`pto.vmod` MUST 通过 internal soft helper path 实现，而不是要求新的 VPTO public op。  
helper call MAY 出现在 frontend materialized module 中，但它 MUST 通过既有 backend-inline 主线在后续阶段被消除。

#### Scenario: integer `pto.vmod` lowers through helper plus backend-inline

- **WHEN** 用户以当前支持的整数 family 调用 `pto.vmod`
- **THEN** frontend MUST 生成合法的 helper-based lowering 形态
- **AND** 后续 backend 主线 MUST 消除对应 helper call

### Requirement: integer 8-bit div/mod MUST be implemented through an explicit widen / narrow profile

对于 `i8/u8` family，本 change 的 `vdiv` / `vmod` 支持 MUST 通过明确的 widen / narrow soft path 完成。  
也即，实现 MUST 先把 8-bit lane 扩展到更宽整数 profile，完成 soft `div/mod`，再收敛回 8-bit 结果。  
该 widen / narrow 过程属于实现细节，但其存在本身 MUST 被视为正式 lowering contract 的一部分。

#### Scenario: `i8` vector div/mod does not depend on a fictitious direct 8-bit hardware op

- **WHEN** 用户对 `i8` 向量调用 `pto.vdiv` 或 `pto.vmod`
- **THEN** lowering MUST 走正式定义的 widen / soft-compute / narrow 路线
- **AND** MUST NOT 假定存在可直接承载该语义的 8-bit hardware `vdiv` / `vmod` op
