# Proposal: 用 `pto.vdiv` / `pto.vmod` 收敛 TileLang DSL 的软实现细节

## 概述

当前仓库中已经存在整数 `vdiv` / `vmod` 的软实现算法，但这套实现仍以 `lib/TileOps/math.py` 中的 helper 形式存在，尚未完全收敛到 TileLang DSL 的正式 public surface。  
与此同时，`pto.vdiv` 在 DSL 侧已经是公开 API，但其 lowering 仍缺少按 dtype 分流的稳定契约：`f16/f32` 需要继续走现有 VPTO `vdiv` 指令路径，`i8/i16/i32` 需要改走内部软实现路径，而不把软实现 helper 暴露给用户。

本 change 的目标是把 `vdiv` / `vmod` 的用户心智统一收敛为：

- 用户只写 `pto.vdiv(...)` / `pto.vmod(...)`
- `f16/f32` `vdiv` 继续保留硬件/VPTO 指令 lowering
- 整数 `vdiv` 与 `vmod` 通过内部 soft helper lowering
- 软实现 helper 不作为 TileLang DSL public API 暴露

## 背景与动机

当前实现存在四个直接问题：

1. `pto.vdiv` 已经是 public surface，但没有冻结“浮点走 VPTO、整数走 soft path”的 lowering 契约。
2. `vmod` 目前只有内部 soft helper，没有完整的 `pto.vmod` public surface 和 end-to-end lowering 链路。
3. 现有 soft helper 仍以可见名字存在，容易把实现细节泄漏给 DSL 用户。
4. `i8` 族虽然已经被当成 `pto.vdiv` 的支持范围之一，但当前 soft helper 还没有形成明确、可验证的 `i8/u8 -> widen -> div/mod -> narrow` 契约。

如果不把这几层契约一次性写清楚，后续实现很容易出现：

- 文档支持范围与真实 lowering 分叉
- 用户直接依赖内部 helper 名字
- `f16/f32` 与整数 `vdiv` 走出两套无 spec 约束的隐式路径
- `vmod` 继续停留在“有算法、无 public surface”的半完成状态

## 目标

- 把 TileLang DSL 的除法/取模 public surface 统一为 `pto.vdiv` / `pto.vmod`。
- 明确 `pto.vdiv` 的 dtype-directed lowering：
  - `f16/f32` 走 authoring-form VPTO `pto.vdiv`
  - `i8/i16/i32` 走内部 soft helper
- 为 `pto.vmod` 补齐完整 public surface 和内部 soft lowering 链路。
- 规定 soft helper 只作为内部实现细节存在，不作为用户可依赖 API 暴露。
- 把 `i8` 族 soft div/mod 的 widen / narrow 路径纳入正式实现范围。

## 非目标

- 不在本 change 中扩展 `bf16` 的 `vdiv` / `vmod` 支持。
- 不在本 change 中重新定义 floating-point remainder/fmod 语义；`pto.vmod` 本次优先收敛整数族 public surface。
- 不在本 change 中引入新的 public helper 命名空间或让用户显式调用 soft helper。
- 不在本 change 中改变现有 `inline_proc` backend-inline 主线；本 change 仅复用该能力承载内部 soft helper lowering。

## What Changes

- `tilelang-dsl-surface`：
  - 明确 `pto.vdiv` 是唯一公开的 vector division surface，支持 `i8/i16/i32/f16/f32`。
  - 新增 `pto.vmod` 作为公开 vector modulo surface，优先覆盖整数族。
  - 明确用户不得依赖内部 soft helper 名字。
- `tilelang-dsl-diagnostics`：
  - 明确 `pto.vdiv` / `pto.vmod` 的 dtype reject 行为。
  - 明确内部 soft helper 名字不属于 public DSL call surface。
- `tilelang-dsl-vpto-lowering`：
  - 为 `pto.vdiv` 新增 dtype-directed lowering 契约。
  - 为 `pto.vmod` 新增“public call -> internal helper -> backend-inline -> legal VPTO”契约。
  - 明确整数 `i8` 族 soft path 需要通过 widen / narrow profile 实现。

## Capabilities

### New Capabilities

- 无

### Modified Capabilities

- `tilelang-dsl-surface`: 统一 `vdiv` / `vmod` 的 public API，隐藏 soft helper 实现细节。
- `tilelang-dsl-diagnostics`: 明确 `vdiv` / `vmod` 的支持范围与内部 helper reject 行为。
- `tilelang-dsl-vpto-lowering`: 为 `vdiv` 增加 dtype-directed lowering，并为 `vmod` 补齐 public-to-soft-lowering 链路。

## 预期结果

- DSL 用户只需要面向 `pto.vdiv` / `pto.vmod` 编程，不再接触 `math.py` 中的 soft helper 名字。
- `f16/f32` `vdiv` 继续保留现有 VPTO 指令 authoring/lowering 路径。
- 整数 `vdiv` / `vmod` 通过内部 soft helper 统一落到 backend-inline 主线，不把软实现细节暴露为 public contract。
- `i8` 族 support matrix 和实现路径重新一致，不再只有表层放行而无清晰 lowering 契约。

## 成功标准

- 新增 `openspec/changes/hide-soft-divmod-behind-pto-surface/`，包含 `proposal.md`、`design.md`、`tasks.md`。
- 新增 spec delta：
  - `specs/tilelang-dsl-surface/spec.md`
  - `specs/tilelang-dsl-diagnostics/spec.md`
  - `specs/tilelang-dsl-vpto-lowering/spec.md`
- 变更文本明确写清：
  - `pto.vdiv` 的双 lowering 路径
  - `pto.vmod` 的 public surface 与 soft lowering
  - soft helper 的 internal-only 定位
  - `i8` 族 widen / narrow soft path 要求

## Impact

- 受影响目录：
  - `tilelang-dsl/python/tilelang_dsl/`
  - `tilelang-dsl/tests/`
  - `tilelang-dsl/docs/user_guide/`
  - `lib/TileOps/`
  - `openspec/changes/hide-soft-divmod-behind-pto-surface/`
- 受影响 public API：
  - `pto.vdiv`
  - `pto.vmod`
- 受影响 lowering 行为：
  - `pto.vdiv` 的 dtype-directed lowering
  - `pto.vmod` 的 internal soft-helper lowering
