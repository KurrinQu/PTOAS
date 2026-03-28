# Proposal: 用 A5VM backend 替换 ptoas 中旧的 A5 OP-Lib lowering 主线

## 概述

当前 `ptoas.cpp` 中 A5 路径仍然以旧的 OP-Lib lowering 主线组织 pass pipeline，`PTOInstantiateAndLowerToLibCall` / `PTOInlineLibCall` / 旧形态 `PTOLowLevelLoopFusion` 与新的 `createLowerPTOToA5VMPass()` 并存，导致 backend 语义边界不清晰。随着 A5VM lowering 已经进入仓库并成为新的 A5 后端实现，`ptoas` 需要把 A5 主线明确切换到 `PTOToA5VM`，并把 tile fusion 的前后段 pass 边界重新稳定下来。

## 背景与动机

当前仓库里已经同时存在两套 A5 后端思路：

- 旧路径：`PTOFusionRegionGen -> PTOViewToMemref -> PlanMemory/InsertSync -> OP-Lib lowering -> inline -> low-level loop fusion -> EmitC`
- 新路径：共享前端 pass 之后进入 `createLowerPTOToA5VMPass()`，再由 A5VM text / HIVM emitter 输出

但 `tools/ptoas/ptoas.cpp` 还保留着以 OP-Lib 为中心的 A5 分流逻辑，带来三个直接问题：

1. A5 主线的真实 backend 边界不清楚，`EmitC`、OP-Lib、A5VM 三者在一个入口里交错。
2. `PTOFusionRegionGenPass`、`PTOToA5VM`、`PTOLowLevelLoopFusion` 的顺序没有被 spec 锁定，后续调整容易再次回退到旧链路。
3. 当前 `PTOLowLevelLoopFusion` 仍按旧 `pto.simd.vec_scope` / `vector.masked_*` 形态定义，而新主线希望它在 `PTOToA5VM` 之后对 A5VM 低层 loop 结构生效。

如果不先把 `ptoas` 的 A5 backend 主线和 tile fusion pass 边界写成正式 contract，后续实现很容易继续在 `ptoas.cpp` 里叠加条件分支，使 A5VM backend 再次退化为“旁路选项”而不是正式 backend。

## 目标

- 将 `ptoas` 中 A5 主线的旧 OP-Lib lowering 路径从默认 backend pipeline 中移除，改为以 `createLowerPTOToA5VMPass()` 为核心的 A5 lowering 路径。
- 固定 `PTOFusionRegionGenPass` 在 `PTOToA5VM` 之前执行，确保 fusion region 先完成结构化封装，再进入 A5VM backend lowering。
- 固定 `PTOLowLevelLoopFusion` 在 `PTOToA5VM` 之后执行，并把它的契约改写为 A5VM post-lowering low-level fusion，而不是旧的 OP-Lib/vector bridge 专用 pass。
- 保留 `EmitC` 作为可选输出路径，但不再让 A5 `EmitC` 依赖 OP-Lib lowering 主线。
- 为 `ptoas.cpp` 的 backend 分流和相关 regression 提供 OpenSpec 级别的稳定约束。

## 非目标

- 不在本 change 中删除仓库里所有 OP-Lib 相关代码或现有 `oplib/` 目录资产。
- 不要求一次性重写所有 A5VM lowering family 的实现细节；本 change 只定义 `ptoas` backend pipeline 与 pass 边界。
- 不改变 A3 的主线 pipeline。
- 不在本 change 中引入新的用户可见 CLI 模式，只重定义现有 A5 backend/fusion 相关选项的行为边界。

## What Changes

- **BREAKING**: `ptoas` 的 A5 主线不再通过旧的 `PTOInstantiateAndLowerToLibCall` / `PTOInlineLibCall` OP-Lib lowering 链接入最终 backend。
- A5 backend 主线改为 `PTOFusionRegionGen -> PTOToA5VM -> PTOLowLevelLoopFusion(A5VM)` 的正式顺序。
- `PTOLowLevelLoopFusion` 的合同从旧 `pto.simd.vec_scope` / `vector.masked_*` 形态迁移到 `PTOToA5VM` 产出的 A5VM 低层 loop 结构。
- `EmitC` 路径继续保留，但不再依赖旧 OP-Lib lowering 管线来表达 A5 backend。
- `ptoas.cpp` 中的 A5 backend 分流、`--enable-op-fusion` 行为边界和 `fusion_region` 生命周期将被重新整理并锁定。

## Capabilities

### New Capabilities

- `a5vm-backend-pipeline`: 定义 `ptoas` 中 A5 backend 应如何通过 `PTOToA5VM`、A5VM post-lowering loop fusion 和最终 A5VM/HIVM 发射组织 pass pipeline。

### Modified Capabilities

- `tile-fusion-region-lowering`: 将 `pto.fusion_region` 的下游 lowering 主线从旧 OP-Lib lowering 更新为 `PTOToA5VM -> PTOLowLevelLoopFusion -> PTOFlattenFusionRegion`。
- `tile-fusion-region-encapsulation`: 补充 `PTOFusionRegionGenPass` 必须先于 A5 backend lowering 运行，并作为进入 `PTOToA5VM` 前的稳定结构化边界。

## 预期结果

- `tools/ptoas/ptoas.cpp` 的 A5 backend 组织方式变成清晰的两段：共享前端/fusion 封装段，和 A5VM backend lowering/emit 段。
- `PTOFusionRegionGenPass` 的位置不再依赖临时实现细节，而是稳定地位于 `PTOToA5VM` 之前。
- `PTOLowLevelLoopFusion` 在新主线里明确成为 A5VM lowering 之后的低层优化，而不是旧 OP-Lib/vector bridge 的遗留 pass。
- A5 `EmitC` 若继续保留，也不再被迫穿过旧 OP-Lib lowering 路径。

## 成功标准

- OpenSpec 中存在一个新的 `a5vm-backend-pipeline` capability，明确 A5 backend 主线以 `PTOToA5VM` 为核心。
- OpenSpec 中 `tile-fusion-region-lowering` 明确要求 `PTOFusionRegionGen` 之后进入 `PTOToA5VM`，并在 `PTOToA5VM` 之后运行 `PTOLowLevelLoopFusion` 与显式 flatten。
- OpenSpec 中 `tile-fusion-region-encapsulation` 明确 `PTOFusionRegionGenPass` 是 A5 backend lowering 之前的正式结构化边界。
- 该 change 对实现方形成无歧义约束：不再把旧 OP-Lib lowering 视为 `ptoas` A5 主线的正式 backend 路径。

## Impact

- 主要影响代码：
  - `tools/ptoas/ptoas.cpp`
  - `lib/PTO/Transforms/PTOToA5VM.cpp`
  - `lib/PTO/Transforms/TileFusion/PTOLowLevelLoopFusion.cpp`
- 主要影响的 pass/接口：
  - `createPTOFusionRegionGenPass()`
  - `createLowerPTOToA5VMPass()`
  - `createPTOLowLevelLoopFusionPass()`
  - `createPTOFlattenFusionRegionPass()`
- 主要影响的测试面：
  - A5VM backend wiring / debug IR
  - tile fusion region lifecycle
  - `ptoas` A5 backend pass 顺序回归
