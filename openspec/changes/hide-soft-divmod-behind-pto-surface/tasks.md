## 1. OpenSpec 契约落定

- [x] 1.1 完成 `specs/tilelang-dsl-surface/spec.md`，固定 `pto.vdiv` / `pto.vmod` public surface 与 internal helper 不外露契约。
- [x] 1.2 完成 `specs/tilelang-dsl-diagnostics/spec.md`，固定 `vdiv` / `vmod` dtype reject 与 internal helper name reject 契约。
- [x] 1.3 完成 `specs/tilelang-dsl-vpto-lowering/spec.md`，固定 `vdiv` 双路径 lowering、`vmod` soft lowering 与 `i8` widen / narrow 约束。

## 2. Frontend / semantic surface

- [x] 2.1 在 `tilelang-dsl/python/tilelang_dsl/semantic.py` 中为 `pto.vdiv` 增加 dtype-directed rewrite：`f16/f32` 保持 `pto.vdiv`，整数族改写为 internal helper call。
- [x] 2.2 为 `pto.vmod` 补齐 public surface、semantic 分析与 dtype 校验。
- [x] 2.3 增加 internal soft helper 注入机制，使 kernel 无需显式 import helper 即可完成 rewrite。
- [x] 2.4 保持 internal helper 名字不属于 public DSL call surface。

## 3. Soft helper implementation

- [x] 3.1 整理 `lib/TileOps/math.py` 中现有 `vdiv` / `vmod` soft helper，使其 internal-only。
- [x] 3.2 补齐 `i8/u8` 的 soft `vdiv` 路径，采用 widen -> div -> narrow profile。
- [x] 3.3 补齐 `i8/u8` 的 soft `vmod` 路径，采用 widen -> mod -> narrow profile。
- [x] 3.4 明确整数 `vdiv` / `vmod` 的除零返回约定与符号约定，并在测试中锁定。

## 4. Lowering / backend path

- [x] 4.1 确保 `f16/f32` `pto.vdiv` 继续走现有 authoring-form VPTO / backend emitter 路径。
- [x] 4.2 确保整数 `pto.vdiv` / `pto.vmod` 通过 internal helper + backend-inline 收敛，不把 helper 名字暴露为最终 public contract。
- [x] 4.3 为 `pto.vmod` 路径补齐与现有 inline-proc backend-inline 主线的接线验证。

## 5. 回归测试与文档

- [x] 5.1 更新 `tilelang-dsl/tests/test_tilelang_dsl_v1.py`，区分浮点 `vdiv` VPTO path 和整数 `vdiv` helper path。
- [x] 5.2 新增 `pto.vmod` 的 public surface 正向/负向测试。
- [x] 5.3 新增 `i8` 族 `vdiv` / `vmod` regression，锁定 widen / narrow 行为。
- [x] 5.4 更新 `tilelang-dsl/docs/user_guide/11-vector-arithmetic-operations.md`，明确 `vdiv` / `vmod` public contract 与 dtype 支持范围。

## 6. 验证

- [x] 6.1 执行针对 `vdiv` / `vmod` 的 TileLang DSL 单测。
- [x] 6.2 执行覆盖 helper + backend-inline 收敛路径的定向回归。
- [x] 6.3 执行 `openspec validate hide-soft-divmod-behind-pto-surface --type change --strict --json --no-interactive`。
