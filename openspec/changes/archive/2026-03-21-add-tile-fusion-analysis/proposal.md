## Why

### 概述

当前仓库里还没有一个可复用的 tile fusion 分析层。现有 `PTOCreateFusionGroups` 直接在 IR 上做线性链分组，既不能表达 `paged_attention_example_kernel_online_update.pto` 中的真实热点子图，也无法作为后续 `FusionPlanPass` 和 `OpSchedulingPass` 的稳定输入。要把 tile fusion 做成独立能力，第一步必须先补齐 5.1 的 `PreFusionAnalysisPass`。

### 背景与动机

这一阶段的 tile fusion 工作已经明确限定在 `tile_buf world`，且希望全局依赖分析尽量同时支持 SSA 和 DPS 两种输入形态。在这个边界下，现有实现存在 4 个基础缺口：

1. 缺少统一的 DFG / 生命周期视图

- 现在没有一个分析结果能统一回答“谁生产了这个 tile”“谁最后消费了这个 tile”“两个 OP 是否处于同一迭代域”。
- 这导致后续策略和调度都只能在 pass 内做局部、临时且不可复用的判断。

2. SSA 和 DPS 还没有统一语义层

- 设计目标是让 planning 层同时适配 SSA 与 DPS，而不是强依赖 `PTOConvertToDPS`。
- 当前仓库没有一个抽象层能把两种输入形态规约到同一套 producer / consumer 语义。

3. `treshape` 语义尚未被单独建模

- `treshape` 不是普通 compute op，但也不是全局屏障。
- 这阶段需要的语义是：`OPA -> treshape -> OPB` 这条依赖链不参与融合判断，但和它无关的其他 op 不能被 `treshape` 无脑阻断。

4. 动态 shape 保守性没有被明确契约化

- 5.2 暂不实现，因此依赖动态符号等价证明的 case 需要在 5.1 就有明确的“不可证即保守”的输出语义。

## What Changes

本 change 明确只覆盖 `docs/tile_fusion/tile_fusion_design_spec.md` 中的 5.1 `PreFusionAnalysisPass`。
它不包含 5.3 的分组/planning，也不包含 5.4 的调度/scheduling。

### 目标

- 新增 `PreFusionAnalysisPass`，为 tile fusion planning 提供独立分析结果。
- 明确该 pass 工作在 `tile_buf world`，并且设计上同时支持 SSA / DPS 两种输入形态。
- 输出统一的分析信息，至少包括：
  - block-local DFG
  - tile 生命周期区间
  - 迭代域等价类
  - hard boundary / local boundary 分类
- 将 `treshape` 建模为局部非穿透边界：
  - `OPA -> treshape -> OPB` 不形成可穿透的融合依赖
  - 与其无关的 op 不因 `treshape` 被整体阻断
- 明确 v1 的保守规则：凡是依赖 5.2 才能证明的动态 shape 相等关系，一律输出为不可证。

### 非目标

- 不在本 change 中实现分组/planning pass。
- 不在本 change 中实现调度/scheduling pass。
- 不在本 change 中产生 `fusion_id` 或组内顺序标签。
- 不在本 change 中做物理重排、OP 聚拢或 `fusion_region` 封装。
- 不实现 5.2 `ShapeInferencePass`。
- 不触及 5.5 之后的 materialization / inline / low-level fusion / load-store elimination。
- 不引入新的用户可见 CLI 或公开 IR op。

### 预期结果

- 仓库内出现一个独立的 tile fusion 分析层，后续 `FusionPlanPass` 和 `OpSchedulingPass` 直接复用其结果。
- `PreFusionAnalysisPass` 可以在不强依赖 `PTOConvertToDPS` 的前提下，同时接受 SSA / DPS 输入。
- `treshape` 的语义边界被明确固定，避免后续规划阶段重复争论它究竟是 relay、硬边界还是普通 compute op。

## Capabilities

### New Capabilities

- `tile-fusion-analysis`: 约束 `tile_buf world` 中的全局依赖分析能力，覆盖 SSA / DPS 双输入、DFG / 生命周期 / 迭代域输出，以及 `treshape` 的局部非穿透语义。

### Modified Capabilities

- 无

## Impact

### 预期影响

- 受影响源码主要包括 `include/PTO/Transforms/Passes.h`、`include/PTO/Transforms/Passes.td`、`lib/PTO/Transforms/` 下的新分析实现，以及 `test/tile_fusion/`。
- 后续 planning / scheduling change 将以该分析结果作为直接前置依赖。
- 这次 change 以分析结果为主，不要求改变默认代码生成结果。

### 成功标准

- OpenSpec 中新增 `tile-fusion-analysis` capability，清晰定义分析输出和边界语义。
- 变更完成后，`PreFusionAnalysisPass` 能对 SSA / DPS 输入给出一致的 DFG 与生命周期视图。
- 变更完成后，`treshape` 被稳定地视为局部非穿透边界，而不是全局硬屏障。
- 变更完成后，动态 shape 不可证 case 会稳定输出保守结果，而不是在后续策略阶段临时兜底。
