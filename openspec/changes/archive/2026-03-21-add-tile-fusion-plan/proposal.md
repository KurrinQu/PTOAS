## Why

### 概述

即便有了独立分析层，当前仓库仍缺少一个真正面向 DAG 子图的融合规划 pass。现有 `PTOCreateFusionGroups` 只覆盖少量线性链，无法表达 `online_update` 样例中的 diamond 和 join 热点。因此第二个独立 change 需要专门落 5.3 的 `FusionPlanPass`。

### 背景与动机

`FusionPlanPass` 的职责不是做后段 lowering，而是把 5.1 的结构化分析结果转成稳定的融合组和组内顺序元数据。当前实现至少有 4 个问题：

1. 分组算法仍然是线性链模型

- 现有规则只检查“当前 op 是否消费前一 op 的 dst”。
- 这无法覆盖 `online_update` 的 1x16 diamond 和 16x128 join 子图。

2. 当前 allowlist 仍围绕旧的 12 个 op

- 驱动样例还需要 `texp`、`texpands`、`trowexpandmul`、`trowexpanddiv`。
- 如果不把这些纳入 planning 范围，后续实现无法接近真实目标。

3. `treshape` 边界语义需要被真正吃进分组逻辑

- `treshape` 不应该让 `OPA -> treshape -> OPB` 被当作可穿透链路。
- 但它也不应该阻断和它无关的其他候选组。

4. 组内顺序契约还没有被显式定义

- 后续调度需要一个稳定、合法的 `fusion_id` / `fusion.order`。
- 如果这一步不在 planning 阶段固定，调度实现会被迫重新决策。

## What Changes

### 目标

- 新增 `FusionPlanPass`，直接消费 `PreFusionAnalysisPass` 的结果。
- 将分组单元从线性链升级为 block-local、region-local 的 DAG 子图。
- 扩展 planning 范围到当前 driver sample 的最小闭包：
  - 现有 12 个 binary / binary-scalar elementwise
  - `texp`
  - `texpands`
  - `trowexpandmul`
  - `trowexpanddiv`
- 明确 `treshape` 只阻断经过它的依赖链，不作为全局规划屏障。
- 为每个选中的组生成稳定的：
  - `pto.fusion.group_id`
  - `pto.fusion.order`
- 上述输出只定义 5.3 planning 契约，不负责把组成员物理聚拢成连续片段。

### 非目标

- 不做指令物理重排；该职责留给 `OpSchedulingPass`。
- 不实现 5.2 `ShapeInferencePass`。
- 不覆盖 5.5 及之后的 region 封装、materialization 和后段优化。
- 不引入新的用户可见 CLI。

### 预期结果

- 仓库内出现一个真正基于 DFG 的融合规划 pass，而不是扩展版线性链分组器。
- `online_update` 样例中的两个主热点能在 planning 阶段稳定成组。
- 后续调度不需要重新判断“谁该在同一组里”，只需消费 `FusionPlanPass` 的元数据做 5.4 物理聚拢。
- 本 change 到此为止，不覆盖 `OpSchedulingPass` 本身的实现与行为契约。

## Capabilities

### New Capabilities

- `tile-fusion-planning`: 约束 `FusionPlanPass` 的 DAG 分组、op 覆盖范围、`treshape` 局部非穿透语义，以及 `fusion_id` / `fusion.order` 输出契约。

### Modified Capabilities

- 无

## Impact

### 预期影响

- 受影响源码主要包括 `include/PTO/Transforms/Passes.h`、`include/PTO/Transforms/Passes.td`、`lib/PTO/Transforms/` 下的规划 pass、`tools/ptoas/ptoas.cpp` 以及 `test/tile_fusion/`。
- 现有 `PTOCreateFusionGroups` 的角色会被 `FusionPlanPass` 取代或下沉为兼容壳层。
- 后续 `OpSchedulingPass` 将以本 change 输出的 metadata 作为唯一输入。

### 成功标准

- OpenSpec 中新增 `tile-fusion-planning` capability，并明确 DAG 分组契约。
- 变更完成后，`FusionPlanPass` 能稳定识别 `online_update` 的 diamond 和 join 主热点。
- 变更完成后，`treshape` 的局部非穿透语义被正确吃进分组逻辑。
- 变更完成后，输出的 `pto.fusion.group_id` / `pto.fusion.order` 足以直接驱动后续调度。
