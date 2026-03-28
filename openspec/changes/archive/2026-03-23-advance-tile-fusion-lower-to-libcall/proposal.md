## Why

### 概述

当前 tile fusion 主线在 5.5 已经输出 `pto.fusion_region`，但后续 `PlanMemory` / `PTOInsertSync` / `PTOInstantiateAndLowerToLibCall` 之间仍没有一致的 region 消费契约。结果是 `pto.fusion_region` 既不能稳定穿过当前 memref-world `PlanMemory` / `InsertSync`，也没有在 LowerToLibCall 之后被正式消解，导致 tile fusion 主线卡在 5.5 之后。

### 背景与动机

目标形态是让 `PlanMemory` / `InsertSync` 运行在 tile_buf world，并让 `LowerToLibCall` 在其后继续消费 tile_buf-world 的 `pto.fusion_region`。但当前仓库还停留在 `PlanMemory` / `InsertSync` 运行于 memref world 的阶段。

在这个现实前提下，当前实现存在两个直接阻塞：

1. `PTOFusionRegionGenPass` 已经把合法 fusion group 封装成 `pto.fusion_region`，但 `PlanMemory` / `PTOInsertSync` 仍按“普通 block 内线性 op”建模，遇到 region wrapper 会失去分析边界，甚至在 local buffer 场景直接失败。
2. `PTOInstantiateAndLowerToLibCall` 已能在 memref-world 下重写 region 内 compute op，但 grouped path 仍未把 `pto.fusion_region` 当成正式 lowering unit；同时 region 在 inline / low-level fusion 之后也没有统一 flatten 出口，最终会残留到 Emit/手工降级前。

如果现在直接把 `LowerToLibCall` 前移到 `PlanMemory` 之前，确实更接近最终 tile_buf-world 目标，但会迫使 `PlanMemory` / `InsertSync` 立即适配 OP-Lib call / inline 后 IR，改造面更大，且和最终“InsertSync 之后再 LowerToLibCall”的主线边界并不一致。更稳妥的过渡方案，是先补齐 `pto.fusion_region` 在现有 memref-world 主线里的后段消费契约，再把 tile_buf-world 迁移留到下一步。

## What Changes

### 目标

- 为 `pto.fusion_region` 补齐 5.5 之后的正式下游契约，覆盖 `PlanMemory`、`PTOInsertSync`、`PTOInstantiateAndLowerToLibCall`、`PTOInlineLibCall`、`PTOLowLevelLoopFusion` 与显式 flatten 出口。
- 保持当前大阶段顺序不变，不把 `LowerToLibCall` 前移到 `PlanMemory` 之前；优先让 `PlanMemory` / `InsertSync` 对 `pto.fusion_region` 透明。
- 让 grouped `PTOInstantiateAndLowerToLibCall` 直接消费 `pto.fusion_region`，而不是继续依赖 5.5 之前的 per-op `pto.fusion.group_id` / `pto.fusion.order`。
- 将 tile fusion 推进到 LowerToLibCall 时所需的 prelude lowering scope 纳入正式契约，首批覆盖当前热点链路所需的 `trowexpandmul`。
- 新增 `PTOFlattenFusionRegionPass` 或等价显式消解步骤，确保进入 Emit/手工降级前不再残留 `pto.fusion_region`。
- 补齐 focused regression，显式约束：
  - `pto.fusion_region` 不会再让 `PlanMemory` / `InsertSync` 因 wrapper 失败；
  - region 内 supported compute op 能被完整 LowerToLibCall；
  - partially-supported region 会 deterministic hard-fail；
  - Emit 前不会残留 `pto.fusion_region`。

### 非目标

- 不在本 change 中把 `PlanMemory` / `InsertSync` 迁回 tile_buf world。
- 不回退 5.5 `pto.fusion_region` 契约，也不恢复 helper-based `PTOOutlineFusionGroups` 作为 tile fusion 主线输出。
- 不在本 change 中引入 chain-level monolithic OP-Lib template；仍保持“逐 op LowerToLibCall + inline 后低层融合”的路径。
- 不一次性扩展所有 prelude / broadcast / reduction / transpose family；本 change 只补齐推进当前 LowerToLibCall 主线所需的首批 scope。
- 不新增用户可见 CLI、公开 IR 入口或新的公开 type。

### 预期结果

- `pto.fusion_region` 能在现有 memref-world `PlanMemory` / `InsertSync` 下稳定存在，不再因为 wrapper op 触发 unknown-op 失败。
- grouped `PTOInstantiateAndLowerToLibCall` 能把 `pto.fusion_region` 作为正式 lowering unit，并在 region 内直接完成 OP-Lib call rewrite。
- 当前热点链路中的 `trowexpandmul -> trowexpandmul -> tadd` 可以完整进入 region-based LowerToLibCall，而不是只 lower 末端 `tadd`。
- `PTOInlineLibCall` 与 `PTOLowLevelLoopFusion` 继续在 region body 内工作，最后通过显式 flatten 消解 region wrapper。
- 进入 Emit/手工降级前，主线 IR 中不再残留 `pto.fusion_region`，同时不丢失 region results / `pto.yield` 所定义的外部可见边界。

## Capabilities

### New Capabilities

- `tile-fusion-region-lowering`: 定义 `pto.fusion_region` 在 5.5 之后到 Emit 之前的下游消费契约，覆盖 `PlanMemory` / `PTOInsertSync` 透明处理、region-based lowering/inlining/fusion 边界，以及显式 flatten 出口。

### Modified Capabilities

- `oplib-lowering`: 扩展 grouped OpLib lowering，使其直接消费 `pto.fusion_region`，并把首批 tile fusion prelude lowering scope 纳入 active grouped path，保持 deterministic hard-fail 边界。

## Impact

### 预期影响

- 受影响 pass/pipeline 主要包括：
  - `tools/ptoas/ptoas.cpp`
  - `lib/PTO/Transforms/PTOPlanMemory.cpp`
  - `lib/PTO/Transforms/InsertSync/PTOInsertSync.cpp`
  - `lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp`
  - `lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp`
  - `lib/PTO/Transforms/PTOInstantiateAndInlineOpLib.cpp`
  - `lib/PTO/Transforms/PTOLowLevelLoopFusion.cpp`
  - 新增的 `PTOFlattenFusionRegionPass`
- 受影响回归主要包括 `test/tile_fusion/` 与 `test/oplib/` 中围绕 `pto.fusion_region`、grouped LowerToLibCall 和 `trowexpandmul` 的用例。
- 受影响 OpenSpec capability 包括现有 `oplib-lowering`，以及本 change 新增的 `tile-fusion-region-lowering`。

### 成功标准

- `fusion_region_interface.mlir` 一类用例在经过 `PTOViewToMemref` 后，不再因 `pto.fusion_region` wrapper 导致 `PlanMemory` 报出 “Unrecognized type of Operation touches local buffer”。
- `PTOInstantiateAndLowerToLibCall` 能在 `pto.fusion_region` 内原位 lower 当前 active region scope，不依赖已被 5.5 清除的 per-op `pto.fusion.*` metadata。
- `trowexpandmul -> trowexpandmul -> tadd` 一类 region 内链路能完整进入 OP-Lib path，而不是留下 raw PTO compute op。
- Emit/手工降级前，IR 中不再残留 `pto.fusion_region`；`pto.yield` / region result 所表达的外部可见边界能够被正确替换回父 block SSA。
