## Why

### 概述

`FusionPlanPass` 只负责逻辑分组，不负责改变指令的物理位置。要让后续 5.5 `fusion_region` 封装成立，还需要一个单独的调度 change，把同组 op 压缩成连续片段。因此第三个独立 change 专门落 5.4 `OpSchedulingPass`。

本 change 以 5.3 `FusionPlanPass` 已经产出的 group metadata 为前提输入，只处理这些既有 group 的物理聚拢，不重新决定哪些 op 成组。

### 背景与动机

当前仓库里没有一个和 tile fusion 规划解耦的调度阶段，这会带来 4 个直接问题：

1. 后续阶段隐含依赖“组成员天然连续”

- 现有 outline / lower 路径默认处理连续链。
- 一旦分组升级到 DAG 子图，必须有独立调度把组成员聚拢。

2. 组内顺序和物理顺序被混在一起

- 如果调度不独立出来，`FusionPlanPass` 就会被迫同时承担“选组”和“移动指令”两件事。

3. `treshape` 需要更细的调度语义

- 它不参与 fusion group。
- 但若与某个 group 无依赖关系，调度应允许该 group 跨过它聚拢。

4. legality 需要明确落在调度层

- 不能跨 SSA 定义、不能跨 side-effect / barrier / 外部 call、不能跨 region / block，这些都属于调度职责而不是分组职责。

## What Changes

### 目标

- 新增 `OpSchedulingPass`，消费 `FusionPlanPass` 产出的 `pto.fusion.group_id` / `pto.fusion.order`。
- 在 basic block 内将同组成员压缩为连续运行片段。
- 保持组内逻辑顺序与 `pto.fusion.order` 一致。
- 允许 group 跨过与之无关的 `treshape` 聚拢。
- 禁止跨越 hard boundary 或违反 SSA / region 合法性。

### 非目标

- 不重新决定谁属于同一 fusion group。
- 不新增、删除、拆分或合并 `FusionPlanPass` 已选出的 group。
- 不改变 CFG。
- 不做 5.5 `fusion_region` 封装或更后段变换。
- 不引入新的用户可见 CLI。

### 预期结果

- 逻辑 group 和物理重排被分成两个清晰阶段。
- 后续 5.5+ 只需消费已经连续化的组成员。
- `treshape` 不再被误当成全局调度屏障。

## Capabilities

### New Capabilities

- `tile-fusion-scheduling`: 约束 `OpSchedulingPass` 的 block-local 物理聚拢行为、合法移动边界，以及与 `treshape` 的调度关系。

### Modified Capabilities

- 无

## Impact

### 预期影响

- 受影响源码主要包括 `include/PTO/Transforms/Passes.h`、`include/PTO/Transforms/Passes.td`、`lib/PTO/Transforms/` 下的调度 pass、`tools/ptoas/ptoas.cpp` 以及 `test/tile_fusion/`。
- 该 change 会首次显式改变 planning 阶段的 IR 指令顺序。

### 成功标准

- OpenSpec 中新增 `tile-fusion-scheduling` capability，并明确调度合法性边界。
- 变更完成后，`OpSchedulingPass` 能把同组成员在 basic block 内聚拢成连续片段。
- 变更完成后，group 可以跨过无关的 `treshape` 聚拢，但不能跨 hard boundary 或违反 SSA 合法性。
