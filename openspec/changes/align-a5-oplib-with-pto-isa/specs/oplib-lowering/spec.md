## ADDED Requirements

### Requirement: A5 manifest status SHALL reflect current sibling `pto-isa` semantics

PTOAS 的 A5 OpLib manifest MUST 以同级 `pto-isa` 当前可确认的语义来源为准，不得继续把已存在公共 API 或 A5 ST 语义证据的 OP 误标为 `deferred`。

#### Scenario: Public API semantic evidence clears stale deferred status

- **WHEN** 某个 4.5~4.9 in-scope OP 在同级 `pto-isa` 的 `include/pto/common/pto_instr.hpp` 中已有稳定公共 API 语义，且 A5 ST 用例可发现对应覆盖，例如 `trecip`
- **THEN** PTOAS A5 manifest MUST 将其视为可实现语义来源的一部分，并更新 `a5_status` / `deferred_reason`，不得继续保留过期的 `deferred` 结论

#### Scenario: Unsupported dtype must not remain implemented in manifest

- **WHEN** 同级 `pto-isa` A5 头文件明确限制某个 OP 的 dtype 集，例如 bitwise family 为整数类型子集
- **THEN** PTOAS A5 manifest MUST 收敛到相同 dtype 集，不得继续把不合法 dtype 标记为 supported

### Requirement: Implemented/deferred classification SHALL distinguish native A5 `_IMPL` gaps from approved decomposition semantics

PTOAS MUST 对 A5 manifest 中的 `implemented` 和 `deferred` 给出一致分类规则，区分“上游已有原生 A5 语义”、“上游仅有公共 API 重写语义”以及“上游仍缺少可采纳语义”。

#### Scenario: Approved decomposition semantics can participate in implemented lowering

- **WHEN** 某个 OP 的同级 `pto-isa` 公共 API 已将其稳定重写为已有 A5 指令语义，且 PTOAS 已明确接受该重写作为 OpLib / lowering 语义来源
- **THEN** manifest 和 lowering MUST 允许该 OP 进入 implemented 集，并按该重写语义选择 template / instance

#### Scenario: Unapproved decomposition remains deterministically deferred

- **WHEN** 某个 OP 虽然存在公共 API 入口，但没有原生 A5 `_IMPL`，且当前 PTOAS 尚未批准其 decomposition 进入 OpLib implemented 集，例如 `taddc`、`tsubc`、`taddsc`、`tsubsc`
- **THEN** manifest MUST 保持 `deferred`，并给出明确 `deferred_reason`

### Requirement: Lowering parity checks SHALL validate manifest-to-template coverage at dtype granularity

PTOAS 的 A5 OpLib lowering 门禁 MUST 不仅检查 implemented op 是否“至少有一个模板”，还必须验证关键 family 的 dtype 覆盖与 manifest implemented 集一致。

#### Scenario: Implemented op missing required dtype coverage fails validation

- **WHEN** manifest 将某个 OP 标记为 implemented，且声明支持特定 dtype
- **THEN** PTOAS 的对齐检查 MUST 能发现 concrete template 对这些 dtype 的缺失，而不是仅因为存在一个无关 dtype 的模板就通过

#### Scenario: Lowering remains hard-fail on manifest/template mismatch

- **WHEN** 用户在 A5 OpLib 流水线中触发某个 manifest-implemented 的 dtype / variant，而 template registry 无法选择候选实例
- **THEN** `PTOLowerToOpLibCalls` MUST 发出确定性错误并失败，不得静默回退到原始 PTO op 或仅记录 warning
