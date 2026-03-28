## Context

### 范围

本 design 只覆盖 5.3 `FusionPlanPass`。

这里的“覆盖”严格指：

- 如何从 5.1 分析结果生成 DAG 融合组
- 如何为组成员写出稳定的 planning metadata

本 design 不覆盖：

- 5.4 `OpSchedulingPass` 的物理重排
- 任何把离散 group member 聚拢成连续片段的算法
- 5.5 及之后的 region / materialization / lowering 行为

其前置条件固定为：

- `PreFusionAnalysisPass` 已存在并能提供 block-local DFG、生命周期与迭代域信息
- 输入仍处于 `tile_buf world`
- 不依赖 5.2

其后置条件固定为：

- 为目标 op 打上 `pto.fusion.group_id`
- 为组内成员打上稳定的 `pto.fusion.order`
- 不改变物理顺序
- 不要求同组成员在 IR 中已经连续相邻

### 当前状态

当前仓库的分组逻辑还停留在 `PTOCreateFusionGroups`：

- 只支持线性连续链
- 主要覆盖旧的 12 个 op
- 不理解 `treshape` 的阶段边界语义
- 没有显式 cost model 和策略接口

### 约束

- 只在单个 basic block 和单个 region 内分组
- 不跨 hard boundary 分组
- 不把 `treshape` 依赖链当作可穿透链路
- 动态迭代域不可证时一律保守拒绝

## Goals / Non-Goals

**Goals:**

- 实现独立的 `FusionPlanPass`
- 支持 diamond / join DAG
- 扩展到 driver sample 需要的最小 op 闭包
- 输出稳定的组 ID 和组内顺序

**Non-Goals:**

- 不做物理调度
- 不做 shape inference
- 不做后段 region / materialization / codegen

## Decisions

### 决策 1：`FusionPlanPass` 直接取代线性链分组语义

本 change 不再沿用“前一 op 的 dst 被后一 op 消费”这一线性分组定义，而是让 `FusionPlanPass` 直接基于 DFG 进行 block-local DAG 分组。

采用该方案的原因：

- 这是支持 `online_update` 主热点的必要条件。
- 如果继续保留线性链为核心语义，后续只会在边角上不断打补丁。

### 决策 2：v1 默认实现一个保守贪心规划器，并预留策略接口

`FusionPlanPass` 提供两层内部接口：

- `StrategyEngine`
- `CostModel`

但 v1 只落一个保守贪心实现。评分因子固定包含：

- 中间 tile 依赖边收益
- 循环控制合并收益
- 活跃 tile 数量惩罚
- VF 参数数量惩罚
- 动态 shape 不可证的强拒绝

采用该方案的原因：

- 与设计文档的可扩展目标一致。
- 同时避免把 v1 复杂度拉到外部插件或 ML 输入层面。

### 决策 3：`treshape` 不进入 group，但也不作为全局 planning 屏障

在 `FusionPlanPass` 中：

- `OPA -> treshape -> OPB` 不建立同组候选关系
- 若 `OPC` 与该 `treshape` 无关，则 `OPC` 与其它合法 op 仍可在同一 planning 区域成组

采用该方案的原因：

- 这精确匹配了当前用户对 `treshape` 的边界定义。

### 决策 4：组内顺序在 planning 阶段固定为稳定拓扑序

`FusionPlanPass` 必须输出稳定的 `pto.fusion.order`，规则为：

- 先满足拓扑依赖
- 再尽量贴近原始顺序

这样做的原因：

- 后续 `OpSchedulingPass` 不需要重新推导“组内逻辑顺序”，只需做物理聚拢。

### 决策 5：planning metadata 是 5.3 与 5.4 之间的唯一契约

本 change 中，`FusionPlanPass` 产出的 `pto.fusion.group_id` / `pto.fusion.order` 是 planning 对 scheduling 的唯一承诺。

这意味着：

- planning 负责“谁在一组、组内逻辑顺序是什么”
- scheduling 负责“是否、以及如何把这些成员物理聚拢”
- 两者不在同一个 change 内混合实现

## Risks / Trade-offs

- [Risk] 过于保守的 cost model 会错过一部分潜在收益。
  → Mitigation：先保证 diamond / join 主热点稳定成组，更多激进策略留给后续迭代。

- [Risk] 扩展 op 范围后，某些 family 的 legality 规则可能不够完整。
  → Mitigation：v1 只覆盖 driver sample 的最小闭包；其余 family 继续拒绝。

- [Risk] `treshape` 的局部边界语义若在实现中处理不细，可能被误当成全局屏障。
  → Mitigation：专门为“无关 op 可跨过 `treshape` 成组”的场景加回归。

## Migration Plan

1. 声明并实现 `FusionPlanPass`。
2. 让其直接消费 `PreFusionAnalysisPass` 的结果。
3. 将 planning 范围扩展到 driver sample 的最小 op 闭包。
4. 用 lit 与 `online_update` driver sample 验证 diamond / join 分组。

## Open Questions

- 现有 `PTOCreateFusionGroups` 在实现阶段是直接删除、重命名，还是保留成一个兼容入口，更利于过渡？
