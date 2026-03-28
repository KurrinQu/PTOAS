## Why

### 概述

当前 tile fusion 路径里已经存在 `PTOOutlineFusionGroups`，但它的输出形态是把融合组 outline 成 `@__pto_fused_group_*` helper 函数。这和 `docs/tile_fusion/tile_fusion_design_spec.md` 中 5.5 的正式目标不一致，5.5 需要的是把已经调度完成的融合片段封装成 `pto.fusion_region` 特殊 region。

本 change 的目标是把 5.5 从“函数 outline”模型切到“region 封装”模型，形成 tile fusion pipeline 的正式结构化 IR 边界。这样后续 5.6+ pass 可以直接消费 region 容器，而不需要继续依赖 helper function 约定。

### 背景与动机

当前仓库在 5.3 `FusionPlanPass` 和 5.4 `OpSchedulingPass` 之后，还缺少一个与设计文档一致的 5.5 封装阶段，这会带来 4 个直接问题：

1. 现有 helper outline 不是 5.5 目标形态

- 它把融合组变成符号函数和 call boundary。
- 但设计文档要求的是 block-local、结构化的 `pto.fusion_region` 容器。

2. 后续阶段缺少正式的 region 语义边界

- helper 函数只能表达“被摘出去的一段代码”。
- 它不能直接表达“这是 tile fusion 的特殊 IR 容器，后续 pass 应以此作为局部优化边界”。

3. 融合组输入输出仍是函数 ABI 语义

- 当前 outline 路径的 interface 是 call args / function args。
- 5.5 更需要显式的 region input / region output / `yield` 语义，以便后续 pass 在同一个函数体内继续工作。

4. 旧路径会模糊阶段职责

- 如果继续把 `PTOOutlineFusionGroups` 当成 5.5 的实现，5.5 与后续 instantiation / inline / loop fusion 的职责边界会持续混乱。
- 需要一个独立 change，把 5.5 的正式 IR 契约固定下来。

## What Changes

### 目标

- 新增 5.5 `PTOFusionRegionGenPass`，消费 5.4 之后已连续化的 fusion group。
- 新增 `pto.fusion_region` 容器 op，用于承载单个 fusion group 的 body region。
- 新增 `pto.yield` terminator op，用于显式返回 region 对外可见的 outputs。
- 把外部输入、外部可见输出、group 身份和 region 闭包约束定义成 PTO dialect 内的正式契约。
- 明确 tile fusion 5.5 的正式输出是 `pto.fusion_region`，不再是 `@__pto_fused_group_*` helper function。

### 非目标

- 不重新决定哪些 op 属于同一 fusion group。
- 不承担 5.4 scheduling 的物理重排职责。
- 不在本 change 中实现 5.6 `PlanMemory/InsertSync`、5.7 `PTOOpLibInstantiationPass`、5.9 `PTOFusionInlinePass` 等后续 consumer 改造。
- 不在本 change 中贯通寄存器前传、loop fusion、load/store elimination 等 5.10+ 优化。
- 不为旧 helper outline 路径补充新能力边界。

### 预期结果

- 5.4 和 5.5 之间形成清晰的阶段契约：5.4 负责让 group 连续，5.5 负责把连续 span 封装成 region。
- tile fusion 主线的中间态从“helper function + call”改为“单函数内的结构化 region”。
- 后续 5.6+ pass 可以直接把 `pto.fusion_region` 作为局部分析与变换边界。

## Capabilities

### New Capabilities

- `tile-fusion-region-encapsulation`: 约束 5.5 `PTOFusionRegionGenPass`、`pto.fusion_region`、`pto.yield` 的结构化封装行为、边界提取规则和闭包合法性。

### Modified Capabilities

- 无

## Impact

### 预期影响

- 受影响源码主要包括 `include/PTO/IR/`、`lib/PTO/IR/`、`include/PTO/Transforms/Passes.h`、`include/PTO/Transforms/Passes.td`、`lib/PTO/Transforms/`、`tools/ptoas/ptoas.cpp` 以及 `test/tile_fusion/`。
- 该 change 会引入新的 PTO dialect region op，并把 tile fusion 5.5 的输出形态从 helper function 改为 region 容器。

### 成功标准

- OpenSpec 中新增 `tile-fusion-region-encapsulation` capability，并明确 5.5 的 region 契约。
- 变更完成后，`PTOFusionRegionGenPass` 能把单个 basic block 内已连续化的 fusion group 封装成一个且仅一个 `pto.fusion_region`。
- 变更完成后，region 的输入、输出和 `yield` 顺序是显式且稳定的，且 region 内不存在未声明的外部 SSA capture。
- 变更完成后，tile fusion 5.5 的正式 OpenSpec 契约不再把 `@__pto_fused_group_*` helper function 作为目标输出。
