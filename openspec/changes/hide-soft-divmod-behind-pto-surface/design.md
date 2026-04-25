## Context

### 范围

本 design 只覆盖 TileLang DSL 中与 `pto.vdiv` / `pto.vmod` 直接相关的 public surface、frontend diagnostics 与 lowering contract：

- `pto.vdiv`
- `pto.vmod`
- internal soft helper 注入与调用建模
- integer `i8/i16/i32` soft div/mod 路径
- `f16/f32` `vdiv` 的 VPTO authoring path

它不覆盖：

- `bf16` div/mod 支持
- floating-point `vmod` / `fmod` 语义扩展
- 新的 backend inline 机制设计
- 非 vector `tdiv*` / `trem*` / `tfmod*` tile op 的独立语义重构

### 当前状态

当前仓库中已经有以下事实：

1. `pto.vdiv` 已是 TileLang DSL public surface，但 lowering 仍按统一 `pto.vdiv` 语义继续往下走，没有在 spec 层冻结 dtype-directed 分流。
2. `lib/TileOps/math.py` 已有整数 `vdiv` / `vmod` soft helper，但它们仍以 helper 形式存在，容易泄漏成用户可见实现细节。
3. `vmod` 还没有完整的 `pto.vmod` public surface / semantic / lowering 链路。
4. `i8` 虽然已被纳入 `pto.vdiv` public support matrix，但当前 soft helper 的正式 contract 还没有覆盖 `i8/u8` widen / narrow profile。
5. 仓库已经具备 `inline_proc` backend-inline 主线，因此“frontend 保留 helper call，backend 主线消除 helper”是可复用路径。

### 设计约束

- 用户编写 DSL 时只能面向 `pto.vdiv` / `pto.vmod`，不能要求显式调用 soft helper。
- `f16/f32` `vdiv` 不能被无差别改写到 soft path；需要保留现有 VPTO 指令 authoring contract。
- `vmod` 本次优先收敛整数族，不在没有明确语义的前提下把 floating `vmod` 一起公开承诺。
- soft helper 的可见性必须是 internal-only，即便其底层继续以 `inline_proc` 或等价 helper 形式存在。

## Goals / Non-Goals

**Goals:**

- 统一 `vdiv` / `vmod` 的用户 surface。
- 让 `vdiv` 在 semantic/lowering 层按 dtype 分流。
- 让 `vmod` 有正式 public surface，但不暴露 soft helper 细节。
- 把 `i8` 族 soft path 补成正式契约的一部分。

**Non-Goals:**

- 不把 `vmod` 定义成新的 VPTO public op。
- 不重新设计整数除零、饱和或 exceptional-value 语义之外的更大算术模型。
- 不让用户通过新的 public helper surface 显式选择“硬件路径”或“软实现路径”。

## Decisions

### 1. `pto.vdiv` 保持单一 public surface，但 lowering 按 dtype 分流

决策：

- `pto.vdiv` 继续是用户唯一可见的 vector division API。
- `f16/f32` `pto.vdiv` 在 frontend/semantic/lowering 中继续保留为 `namespace="pto"` 的 VPTO authoring op。
- `i8/i16/i32` `pto.vdiv` 在 semantic 阶段重写为 internal helper call，再通过 backend-inline 主线消除。

原因：

- 这能保持用户心智统一，同时允许浮点复用现有 VPTO path，整数则走 soft algorithm。

备选方案：

- 让所有 `pto.vdiv` 都走 soft path。
  - 放弃原因：会破坏现有 `f16/f32` VPTO authoring / backend contract，也会丢失已经存在的硬件路径价值。

### 2. `pto.vmod` 作为 public surface 新增，但优先只承诺整数族

决策：

- 用户 surface 新增 `pto.vmod(vec0, vec1, mask)`。
- 本 change 中，`pto.vmod` 的正式支持矩阵优先限定为 `i8/i16/i32` 家族。
- `pto.vmod` 不新增新的 VPTO public op，而是直接走 internal soft helper path。

原因：

- 当前仓库中已经有整数 soft `vmod` 算法，可以形成闭环。
- floating `vmod` / `fmod` 语义尚未被冻结，不适合在本 change 中一起 over-commit。

### 3. soft helper 继续存在，但必须 internal-only

决策：

- soft helper 可以继续以 `inline_proc` 或等价 helper 形式保留。
- helper 命名与接线路径只属于实现细节，不属于 public surface。
- frontend diagnostics 对这些 internal helper 名字仍按“unsupported public call surface”处理。

原因：

- 用户 contract 要以 `pto.vdiv` / `pto.vmod` 为唯一入口，不能让 `math.py` helper 变成事实上的次级 API。

### 4. `i8` 族必须通过 widen / narrow profile 完成 soft div/mod

决策：

- `i8/u8` soft `vdiv` / `vmod` 采用 widen -> soft div/mod -> narrow 的正式实现路线。
- 该 widen / narrow 过程属于实现细节，但其存在本身是本 change 的 contract 一部分。

原因：

- 当前 soft helper 主要覆盖 16/32-bit 家族；如果不把 `i8` 单独写进 contract，就会继续出现“surface 允许、实现无正式路径”的漂移。

## 测试策略

- Python/frontend 单测：
  - `f16/f32` `pto.vdiv` 继续产出 `pto.vdiv`
  - 整数 `pto.vdiv` 产出 internal helper call，而不是 authoring-form `pto.vdiv`
  - `pto.vmod` public surface 的正向/负向测试
  - internal helper 名字 direct call 仍被 reject
- lowering / backend 回归：
  - 验证整数 `vdiv` / `vmod` 通过 helper + backend-inline 收敛
  - 验证 `f16/f32` `vdiv` 保持 VPTO path

## Risks / Trade-offs

- [Risk] `vdiv` 同名但双路径 lowering 可能让测试断言更复杂  
  Mitigation：把测试分成浮点 path 和整数 path 两类，不再只断言统一文本形态。

- [Risk] `vmod` public surface 若过早承诺浮点 family，会让语义边界变模糊  
  Mitigation：本 change 只先冻结整数族，floating `vmod` 另开 change。

- [Risk] internal helper 注入机制会增加 frontend/semantic 复杂度  
  Mitigation：复用已有 inline-proc helper/call 模型，而不是再开第三套 helper 体系。

## Migration Plan

1. 先冻结 OpenSpec contract，明确 `vdiv` / `vmod` 的 public/lowering 边界。
2. 在 frontend/semantic 中补 internal helper 注入与 dtype-directed rewrite。
3. 补齐 `i8` 族 soft div/mod helper。
4. 补测试与文档，验证 `f16/f32` 和整数 path 均符合新契约。

## Open Questions

- 浮点 `pto.vmod` 是否需要在后续 change 中定义为 `fmod` 还是 remainder 语义。  
  本 change 暂不回答该问题。
