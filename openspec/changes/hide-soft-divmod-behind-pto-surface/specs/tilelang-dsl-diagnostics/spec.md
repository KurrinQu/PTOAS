## ADDED Requirements

### Requirement: diagnostics MUST enforce the public `vdiv` / `vmod` support matrix and reject internal helper names

TileLang DSL diagnostics MUST 对 `pto.vdiv` / `pto.vmod` 的 public support matrix 做 fail-fast 校验。  
其中：

- `pto.vdiv` MUST 接受 `i8/i16/i32/f16/f32`
- `pto.vmod` MUST 接受当前公开承诺的整数 family
- `bf16` 和其他未纳入本 change support matrix 的 dtype MUST 报错
- internal soft helper 名字 direct call MUST 继续按 unsupported public call surface 报错

#### Scenario: unsupported `pto.vdiv` dtype is rejected explicitly

- **WHEN** 用户以不在 public support matrix 中的 dtype 调用 `pto.vdiv`
- **THEN** frontend MUST 在生成 IR 之前报错
- **AND** 诊断 MUST 明确指出 `pto.vdiv` 当前支持的 dtype family

#### Scenario: unsupported `pto.vmod` dtype is rejected explicitly

- **WHEN** 用户以不在当前公开承诺范围内的 dtype 调用 `pto.vmod`
- **THEN** frontend MUST 在生成 IR 之前报错
- **AND** 诊断 MUST 明确指出 `pto.vmod` 当前支持范围

#### Scenario: internal soft helper name is not accepted as public DSL surface

- **WHEN** 用户在 TileLang DSL kernel 中直接调用 internal soft helper 名字，而不是 `pto.vdiv` / `pto.vmod`
- **THEN** frontend MUST 把该调用视为 unsupported public call surface
- **AND** 诊断 MUST NOT 暗示用户应该直接依赖该 helper 名字
