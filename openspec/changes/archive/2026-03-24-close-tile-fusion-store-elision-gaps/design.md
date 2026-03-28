## Context

### 当前链路

当前 A5 tile fusion 主线已经是：

`PreFusionAnalysis -> FusionPlan -> OpScheduling -> PTOFusionRegionGen -> PTOViewToMemref -> PlanMemory -> PTOInsertSync -> PTOInstantiateAndLowerToLibCall -> PTOInlineLibCall -> PTOLowLevelLoopFusion -> PTOFlattenFusionRegion -> Canonicalizer/CSE`

从代码和样例 IR 看，冗余 store 残留不是单点问题，而是三个阶段的 contract 没有闭合：

1. `PreFusionAnalysis` 只按底层 `Value` 聚合 liveness

- `lib/PTO/Transforms/TileFusion/FusionAnalysis.cpp` 里 `producerByValue` / `livenessSlotByValue` 都以 `Value` 为 key。
- 在 DPS 复用同一个 destination tile 的场景，后写会覆盖前写，导致 analysis 只能看到“同一 storage 的最后一个 producer”，看不到中间写实例。
- `paged_attention_example_kernel_online_update.pto` 的 1x16 softmax-like 链正是这种情况：
  - `%15` 先后被 `tsub`、`tsub`、`tmul` 复用为 destination。
  - 当前 dump 中 `node1.out0 producer=5`，说明前序写已经被覆盖掉。

2. `PTOFusionRegionGenPass` 需要保持最小化 frontier contract

- `lib/PTO/Transforms/TileFusion/PTOFusionRegionGen.cpp` 的 `buildGroupSpanInterface()` 只看 `hasReplaceableUseOutsideSpan()`。
- 这个判断足以回答“这个值要不要 yield”，并且能够直接定义 region 的正式外部 frontier。
- 对经 `pto.treshape` 暂时离开 region 的值，当前设计决定仍把它们视为普通 yielded frontier；如果未来要对这类 boundary 做进一步融合，应发生在 `FusionRegionGen` 之前，而不是通过扩展 `pto.yield` contract 来完成。

3. lowering / low-level fusion 只覆盖窄模式的 round-trip forwarding

- `PTOLowLevelLoopFusion` 只在相邻 canonical `pto.simd.vec_scope` stage 间做同 base/index/mask 的 `vector.maskedstore -> vector.maskedload` forwarding。
- 它保留“同一个 base 的最后一次 store”，即使这次写已经不再逃逸 region。
- `PTOFlattenFusionRegion` 之后只跑 `Canonicalizer + CSE`，没有专门处理 `vector.maskedstore` 的 dead-store elimination。

### 样例证据

对 `test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto` 的实际跟踪表明：

- `PTOFusionRegionGen` 为第一段 1x16 hotspot 生成 `pto.yield(%14, %16, %17, %19)`。
- `PTOLowLevelLoopFusion` 虽然已经把多个 stage 收敛成单个 `pto.simd.vec_scope`，但 region 内仍保留 6 个 `vector.maskedstore`。
- 最终 `/tmp/kernel_online_update.cpp` 中对应为 6 个 `vsts`：
  - 其中一部分对应 yielded frontier，当前按设计仍会保留。
  - 另一部分对应 region-internal temporary 的最后一次写，本应在 frontier-aware store-elision 中删除，但当前没有实现。

这说明当前问题至少分成两类：

- **internal-dead tail store**：值不再逃逸 region，但最后一次写仍被保留。
- **yielded frontier store**：值仍然通过 `pto.yield` 离开 region；在 v1 中这类写入统一保守保留，无论 region 外后续是否先经过 `pto.treshape`。

### 约束

- 保持 `tile_buf world` 中的 analysis 边界，不把 5.5 之后的 lowering 细节反向耦合进 planning。
- 保持 `pto.treshape` 的 local non-through boundary 语义，不在本 change 中把它改成可直接穿透的 compute op。
- 保持现有 region-preserving lowering 路径，直到显式 flatten 前都以 `pto.fusion_region` / `pto.yield` 作为正式边界。
- 对无法稳定证明的场景保守退化，不做“看起来像 dead store”的激进删除。

## Goals / Non-Goals

**Goals:**

- 为 `PreFusionAnalysis` 补齐 write-instance 级别的 live range 语义，区分“tile storage”与“对该 storage 的一次写”。
- 在 analysis 阶段给每个 write instance 标出 escape kind，至少区分：
  - `internal`
  - `local-boundary-external`
  - `hard-external`
- 让 `PTOFusionRegionGenPass` 把 `pto.yield` frontier 收敛为“稳定顺序的显式 yielded value 列表”，而不是再并行维护额外 frontier metadata。
- 在 lowering 后、flatten 前新增或扩展一个 frontier-aware store-elision 阶段，删除 region 内 non-escaping tail store。
- 保持 `treshape` 留在 fusion region 外；如果后续要对 `treshape` 做特殊融合或 forwarding，应发生在 `FusionRegionGen` 之前。

**Non-Goals:**

- 不改变 `FusionPlanPass` / `OpSchedulingPass` 的组成员决策逻辑。
- 不在本 change 中完成跨 `pto.fusion_region`、跨 `pto.treshape` 的寄存器前传。
- 不尝试把所有 remaining `vector.maskedstore` 都删掉；无法稳定证明时保持保守。
- 不引入新的用户可见 IR op、CLI flag 或架构分歧。

## Decisions

### 决策 1：在 `PreFusionAnalysis` 中引入 write-instance 模型，而不是继续用 `Value` 直接代表 live range

`FusionValueLiveness` 当前只能表达“某个 tile storage value 的整体读写情况”，不足以表达 DPS 复用 destination 时的多次定义。为了解决这一点，analysis 需要显式区分：

- **storage value**：底层 `tile_buf` / `memref` carrier，本质上表示“写到哪里”。
- **write instance**：某个 compute op 对该 storage 的一次逻辑定义，本质上表示“哪次写产生了哪个版本”。

设计上保留现有 block-local DFG / iteration-domain 输出，同时新增 write-instance 级别元数据，例如：

- `writeInstanceId`
- `storageValue`
- `producerNode`
- `consumerNodes`
- `lastLocalConsumer`
- `escapeKind`

这样做的原因：

- `FusionPlanPass` 仍可继续使用现有 node/edge/domain 信息，不需要被 store-elision 需求反向污染。
- `PreFusionAnalysis` 仍然可以保留 escape classification 作为分析输入，但 `PTOFusionRegionGenPass` 不必把这份分类再序列化到 region frontier 上。

替代方案是继续沿用 `Value`-keyed liveness，只修补 `lastLocalConsumer`。该方案无法稳定处理 `%15` 这类多次覆写，因此拒绝。

### 决策 2：`PTOFusionRegionGenPass` 的 frontier 保持为“yielded values”，不再增加 frontier classes

`pto.yield` 的 value 列表继续直接定义 region 对外可见值的正式顺序。本 change 不再为 `pto.yield` / `pto.fusion_region` result 并行生成 frontier-class metadata，原因是：

- `yield` / `not yield` 已经足以表达 v1 store-elision 所需的外部可见边界；
- 经 `pto.treshape` 暂时离开 region 的值，在当前设计中仍按普通 yielded frontier 保守处理；
- 如果未来要把这类 boundary 提前吃进更大的融合区域，应发生在 `FusionRegionGen` 之前，而不是通过扩展 region frontier contract 完成。

之所以不直接在 `PTOFusionRegionGenPass` 中做 store elimination，是因为 5.5 封装阶段仍处于 `tile_buf world`，此时还没有低层 `vector.maskedstore`，只能定义 frontier contract，不能替代 lowering 后的物理消冗。

### 决策 3：保持 planning / scheduling 只消费分析的 DFG 部分，不让 live range 反向改变组成员决策

这次 change 的目标是补齐“冗余 store 消除链路”，不是重写 5.3 / 5.4 的 group legality。为降低风险：

- `FusionPlanPass` / `OpSchedulingPass` 继续使用现有的 node/edge/domain 信息；
- write-instance liveness 与 escape class 保持为分析输入，不让它们反向改变 region frontier 的最小 contract；
- 不在本 change 中重新定义 `treshape` 对 planning 的影响。

这样可以把“成组问题”和“消冗问题”拆开，避免一次 change 同时改 legality 和 codegen。

### 决策 4：新增独立的 `PTOFusionStoreElisionPass`，放在 `PTOLowLevelLoopFusion` 之后、`PTOFlattenFusionRegionPass` 之前

当前 `PTOLowLevelLoopFusion` 负责结构化 loop/stage 合并和窄模式的 forwarding。把 frontier-aware dead-store reasoning 强行塞进这个 pass，会导致两个问题：

- 结构融合与 escape 判定耦合过深，维护代价高；
- 很多“最后一次写是否可删”的判断只有在整个 region lowering 结束后才稳定。

因此本 change 采用独立 pass：

- 输入：memref-world、仍保留 `pto.fusion_region` / `pto.yield` frontier 的 lowered region。
- 位置：`PTOLowLevelLoopFusion` 之后、`PTOFlattenFusionRegionPass` 之前。
- 职责：
  - 删除映射到 non-escaping write instance 的最终 `vector.maskedstore` / 等价 store；
  - 保留所有仍在 yielded frontier 中的 store；
  - 对无法稳定映射回 region frontier / write-instance 的 store 保守保留。

替代方案是只扩展 `PTOLowLevelLoopFusion` 的 `lastStoreStage` 逻辑。该方案仍然只看 stage/base 级别信息，无法表达“同一个 base 的最后一次写虽然没有再被 load，但已经不逃逸”的场景，因此拒绝。

### 决策 5：`treshape` 保持在 fusion region 外，v1 对其后的 yielded frontier 统一保守处理

`online_update` 里 `%16/%17 -> pto.yield -> pto.treshape -> 下游 region` 这类链路说明，当前 remaining store 里有一部分并不是简单 dead store，但本 change 不再试图用 frontier class 区分它们。

本 change 的 v1 范围只要求：

- `treshape` 继续留在 fusion region 外；
- `pto.yield` 只表达显式 yielded frontier；
- store-elision 对所有 yielded frontier 统一保守保留；
- 如果未来需要跨 `treshape` 做 boundary-aware forwarding，应在 `FusionRegionGen` 之前解决。

之所以不在本 change 中直接移除这类 store，是因为跨 `pto.treshape` 的数据前传已经不再是“同 base memref 的 round-trip elimination”，而涉及 view/layout 语义与跨 region 数据传递，复杂度显著更高。

### 决策 6：回归按“分析正确性 / frontier 正确性 / 消冗正确性”三层拆开

测试必须分别锁住三层行为：

1. `test/tile_fusion/pre_fusion_analysis_*.mlir`

- 增加 DPS destination 复用场景；
- 验证 distinct write instance 与 escape class。

2. `test/tile_fusion/fusion_region_*.mlir`

- 验证 `pto.yield` / region result 顺序稳定，且 internal-dead value 不会占用 yielded frontier；
- 验证 internal-dead value 不出现在 frontier 中。

3. `test/tile_fusion/low_level_loop_fusion_*` 或新增 store-elision 用例

- 验证 region 内 non-escaping tail store 被删除；
- 验证 yielded frontier store 仍保留；
- 对 `online_update` 这类 driver sample 增加 focused regression，避免只看 helper 存在而看不到多余 `vsts`。

## Risks / Trade-offs

- [Risk] write-instance 模型会扩大 `PreFusionAnalysis` 结果面，影响现有 dump 与调试习惯
  - Mitigation：保持现有 DFG / domain 输出结构不变，write-instance 信息以增量字段和稳定 dump 文本引入。

- [Risk] 对经 `treshape` 暂时离开 region 的 yielded value 过于保守，导致可删 store 仍然残留
  - Mitigation：v1 接受这部分保守保留；如果后续要进一步优化，改在 `FusionRegionGen` 之前处理 `treshape` 特殊融合，而不是重启 frontier-class 设计。

- [Risk] store-elision 无法把低层 store 稳定映射回某个 region-local write instance，导致可删 store 仍然残留
  - Mitigation：v1 允许保守保留；只要 internal-dead tail store 的主路径被稳定覆盖即可，不追求一次吃掉所有模式。

- [Risk] 用户会把“经 `treshape` 暂时离开 region 的 yielded store 仍保留”误解成 change 失败
  - Mitigation：在 proposal、spec 和测试中明确说明 v1 只区分 yielded / not-yielded frontier；`treshape` 特殊处理属于 region 生成前的后续工作。

- [Risk] 新增独立 pass 会增加流水线复杂度
  - Mitigation：把职责限定在 `pto.fusion_region` 内的 frontier-aware dead-store elimination，不让它承担 group legality、loop fusion 或 boundary-aware forwarding。
