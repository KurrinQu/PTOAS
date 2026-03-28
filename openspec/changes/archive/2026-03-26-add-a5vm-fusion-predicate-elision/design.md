# Design: A5 fusion region 内冗余 `plt` predicate-elision

## Context

当前 A5 fusion backend mainline 在 `tools/ptoas/ptoas.cpp` 中的关键顺序为：

`PTOToA5VM -> PTOLowLevelLoopFusion -> CSE -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion`

仓库内已有两个相邻契约：

- `a5vm-backend-pipeline` 约束 post-lowering mainline 必须发生在 `PTOToA5VM` 之后，并保持 memref-first 地址语义；
- `tile-fusion-store-elision` 约束 store-elision 必须发生在 `pto.fusion_region` / `pto.yield` frontier 仍然显式时。

当前缺口在于：`CSE` 之后 region 内仍残留多组等价 `a5vm.plt_b32`。从现有 `build/run.log` 看，问题不只是“同一 SSA 重复建 op”，还包括 loop `iter_args` 的 loop-carried recurrence：两个 block argument 起点相同，并分别由等价的 `plt.scalar_out` 递推，因此每轮迭代都会再次物化等价 `plt`。通用 `CSE` 不负责这种 region-local、双结果、loop-carried 等价证明。

该变更的约束也很明确：

- 只服务 A5 fusion mainline，不引入全函数通用 A5VM 值传播；
- pass 必须发生在 `pto.fusion_region` 仍显式存在时，避免 flatten 后再从普通 SSA use 反推 recurrence；
- `a5vm.plt_b8` / `plt_b16` / `plt_b32` family 的行为要统一；
- `plt` 是双结果 op，`mask` 与 `scalar_out` 必须按整 op 复用，而不是只做单结果 mask forwarding。

## Goals / Non-Goals

**Goals:**

- 新增一个专门的 `PTOFusionPredicateElisionPass`，作为 `func::FuncOp` pass 接到 A5 fusion 主线内部 `CSE` 之后、`PTOFusionLoadStoreElisionPass` 之前。
- 仅处理 `pto.fusion_region` body 内的 `a5vm.plt_b8`、`a5vm.plt_b16`、`a5vm.plt_b32`。
- 当后一个 `plt` 的 scalar 输入可证明与前一个支配 `plt` 等价时，复用前者的 `mask` 与 `scalar_out`，删除后者。
- 显式支持受控的 loop-carried scalar 等价传播，覆盖当前 `run.log` 中 `%arg11/%arg12` 这类 recurrence。
- 保持与现有 `PTOFusionLoadStoreElisionPass`、`PTOFlattenFusionRegionPass` 的边界清晰分离。

**Non-Goals:**

- 不把该 pass 做成任意 A5VM op 的通用 CSE / GVN。
- 不跨 `pto.fusion_region` 边界建立等价类，也不处理 residual non-fused parent-block A5VM op。
- 不尝试覆盖任意复杂 region/control-flow 的抽象解释；遇到分支、无法证明的 recurrence 或非支持 op 时保持保守。
- 不引入新的 CLI 开关、pass option 或 debug interface。

## Decisions

### 1. 新增独立 pass，而不是扩展 `CSE` 或并入 store-elision

决定：新增 `PTOFusionPredicateElisionPass`，实现放在 `lib/PTO/Transforms/TileFusion/`，并在 `include/PTO/Transforms/Passes.h` / `Passes.td` 注册为独立 pass，pipeline 位置固定在当前内部 `createCSEPass()` 之后。

原因：

- 通用 `CSE` 主要依赖 SSA/等价 op 命中，不承担 loop-carried 值等价证明；
- `PTOFusionLoadStoreElisionPass` 的正式职责是访存往返和 frontier-aware tail-store cleanup，把 `plt` 双结果复用硬塞进去会混淆 capability 边界；
- 单独成 pass 后，可以在 spec 中独立定义输入域、family 范围、递推等价和保守边界，也更便于单独回归。

备选方案：

- 扩展 `CSE`：放弃。无法把 fusion-region-only、loop-carried recurrence 和双结果替换策略限制在当前主线上。
- 合并到 `PTOFusionLoadStoreElisionPass`：放弃。predicate 复用与 store frontier 是两类不同契约，耦合后测试和 spec 都会变混乱。
- 放到 flatten 之后：放弃。届时 `pto.fusion_region` frontier 已消失，不利于限定范围，也更容易误伤非 fusion A5VM IR。

### 2. 按整 op 复用 `plt` 双结果，而不是只做 mask forwarding

决定：把 `a5vm.plt_b8` / `plt_b16` / `plt_b32` 视为一类“必须整体复用”的候选；一旦判定两个 `plt` 等价，就同时把后者的 `mask` 和 `scalar_out` 使用点改写到前者结果，并删除后者。

原因：

- `plt` 的 `mask` 与 `scalar_out` 语义绑定，同一次 lane-count materialization 应该只有一个结果对；
- 只重写 `mask` 会留下孤立的 `scalar_out` producer，既不符合这次需求，也会让后续 recurrence 继续分叉；
- 当前日志中的 loop-carried 情形本质上就是 `scalar_out` 链冗余，必须作为第一等公民处理。

备选方案：

- 仅复用 `mask`：放弃。不能解决 `scalar_out` 递推冗余。
- 仅在 `scalar_out` 无 use 时删除：放弃。会错过最需要处理的 loop-carried 情形。

### 3. 等价判定采用“受控值等价 + loop-carried recurrence 传播”，不做全局抽象解释

决定：pass 在每个 `pto.fusion_region` 内建立一个受控的 scalar 等价模型：

- 基础等价：相同 SSA、由同一支配值经无副作用等价 op 得到的值、或直接来自同一个已接受 `plt.scalar_out` 源；
- recurrence 等价：对支持的 `scf.for`，若两个 iter_arg 的 init 值等价，且上一轮 `scf.yield` 中对应位置都由等价 `plt.scalar_out` 产生，则把下一轮这两个 iter_arg 继续视为等价；
- 候选 `plt` 只有在 op kind 相同、bitwidth 相同、前者支配后者，且 scalar 输入位于同一等价类时才可合并。

实现上不追求通用 SSA solver，而是限定在 fusion-region body 的 `scf.for + a5vm.* + arith.*` 低层形态里进行前向扫描和 recurrence 更新。对无法落入该受控形态的值，一律视为不等价。

原因：

- 该策略刚好覆盖当前 low-level loop fusion 后的 canonical 形态；
- 可以直接承接 `build/run.log` 中的 `%arg11/%arg12` 双 recurrence 问题；
- 保守边界清晰，错误合并风险可控。

备选方案：

- 只按同一 scalar SSA 合并：放弃。解决不了当前日志里的主要冗余模式。
- 复用通用 `ValueEquivalence` / GVN 框架：当前仓库无现成、且范围会超出 change 目标。

### 4. 作用域严格限定在 `pto.fusion_region` body，且只处理 `plt` family

决定：pass 仅扫描 `pto.fusion_region` body，忽略 wrapper 外的 residual non-fused A5VM op；候选 op 只包括 `a5vm::PltB8Op`、`a5vm::PltB16Op`、`a5vm::PltB32Op`。

原因：

- 这是本 change 与 `a5vm-backend-pipeline` / `tile-fusion-region-lowering` 契约对齐后的最小闭包；
- 可以保证 pass 对非 fusion 路径完全无侵入；
- family 边界清楚，便于新增针对 `plt_b8` / `plt_b16` 的回归，而不把其他 predicate op 一并卷入。

备选方案：

- 全函数扫描所有 A5VM op：放弃。会把 non-fused path 也纳入新契约。
- 一并处理 `pset` / `plds` 等其他 mask op：放弃。本次问题与 `plt` 双结果 recurrence 直接相关，先保持 v1 范围收敛。

### 5. 测试以 phase2 IR 回归为主，覆盖 family 正例、loop-carried 正例和保守负例

决定：回归面放在 `test/phase2/`，优先验证 A5VM IR 与 pass 顺序，而不是依赖更重的 sample/NPU 路径。

测试最少包括：

- 一个新的 fusion-region IR dump 用例，检查 `PTOFusionPredicateElisionPass` 之后 region 内重复 `plt_b32` 被折叠；
- 一个直接 A5VM family 用例，证明 `plt_b8` / `plt_b16` / `plt_b32` 都适用同一复用规则；
- 一个负例，证明不同 bitwidth、不同 recurrence 或无法证明等价时，pass 保守保留两个 `plt`；
- 一个 pipeline 顺序检查，确认新 pass 位于 `CSE` 与 `PTOFusionLoadStoreElisionPass` 之间。

原因：

- 该 change 的核心价值是 IR 形态收敛，lit/phase2 是最直接、最稳定的验证面；
- 先把 pass 契约锁住，再决定是否补更重的 sample 覆盖。

## Risks / Trade-offs

- [Risk] 过度合并 loop-carried scalar，错误地把语义不同的 recurrence 视为相等
  - Mitigation: 只在 init 等价、yield 来源都是等价 `plt.scalar_out`、且支配关系成立时传播；一旦出现非支持 op、分支或不同来源即放弃等价。

- [Risk] 新 pass 顺序与 `PTOFusionLoadStoreElisionPass` 相互耦合，后续维护时被误挪动
  - Mitigation: 同时更新 `a5vm-backend-pipeline` spec，并加入显式 pipeline 顺序回归。

- [Risk] `plt_b8` / `plt_b16` 当前在线路中覆盖较少，family-wide 支持可能先暴露测试缺口
  - Mitigation: 用单独 phase2 IR 用例覆盖三种 bitwidth，不把行为只绑定在 `plt_b32` 样例上。

- [Trade-off] 该 pass 不会尝试成为通用 A5VM GVN，因此某些 wrapper 外或复杂 CFG 内的重复 `plt` 仍会保留
  - Rationale: 这是有意的范围约束，优先解决 fusion mainline 中最稳定、最有价值的冗余模式。

## Migration Plan

该 change 只新增内部 pass 与 spec，不涉及用户配置迁移。

实施顺序：

1. 新增 OpenSpec capability 与 pipeline delta，锁定顺序和作用域；
2. 新增 `PTOFusionPredicateElisionPass` 注册与 pipeline 接线；
3. 实现 fusion-region-local `plt` 整 op 复用和 loop-carried 等价传播；
4. 增加 phase2 lit 回归验证正例与负例。

若实现中发现 loop-carried 等价条件不足以安全覆盖某些样例，可先缩窄支持形态，但不得突破当前 spec 中的保守边界。

## Open Questions

当前无额外开放问题。v1 范围已锁定为：fusion-region-only、`plt_b8/b16/b32` family、整 op 双结果复用、支持受控的 loop-carried scalar 等价传播。
