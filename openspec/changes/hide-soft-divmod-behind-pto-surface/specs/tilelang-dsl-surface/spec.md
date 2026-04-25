## ADDED Requirements

### Requirement: TileLang DSL MUST expose `pto.vdiv` / `pto.vmod` as the only public vector div/mod surface

TileLang DSL public surface 中，vector division / modulo MUST 统一通过 `pto.vdiv(...)` 与 `pto.vmod(...)` 暴露。  
用户 MUST NOT 被要求显式调用 `lib/TileOps/math.py` 中的 helper，或依赖任何 internal soft helper 名字来获得整数 `div/mod` 能力。  
`pto.vdiv` MUST 支持 `i8/i16/i32/f16/f32`。  
`pto.vmod` 在本 change 中 MUST 至少支持整数 8/16/32-bit family。

#### Scenario: user writes integer vector division through `pto.vdiv`

- **WHEN** 用户在 TileLang DSL kernel 中编写 `pto.vdiv(lhs, rhs, mask)`，且向量元素类型为 `i8/i16/i32`
- **THEN** frontend MUST 接受该 public surface
- **AND** 用户 MUST NOT 需要额外调用任何 soft helper 名字

#### Scenario: user writes integer vector modulo through `pto.vmod`

- **WHEN** 用户在 TileLang DSL kernel 中编写 `pto.vmod(lhs, rhs, mask)`，且向量元素类型为整数 8/16/32-bit family
- **THEN** frontend MUST 接受该 public surface
- **AND** 该能力 MUST 通过正式 DSL API 提供，而不是停留在内部 helper 层

### Requirement: internal soft helper names MUST NOT become TileLang DSL public API

即便实现层继续保留 `inline_proc` 或等价 helper 作为整数 `div/mod` 的软实现承载，这些 helper 名字也 MUST NOT 成为 TileLang DSL public API。  
用户文档、support matrix 和 surface 说明 MUST 只暴露 `pto.vdiv` / `pto.vmod`，不得把 internal helper 当成推荐或稳定入口。

#### Scenario: public documentation does not advertise soft helper names

- **WHEN** 用户查看 TileLang DSL 的 vector arithmetic public surface
- **THEN** 文档 MUST 只描述 `pto.vdiv` / `pto.vmod`
- **AND** MUST NOT 把内部 soft helper 名字写成用户应直接调用的 API
