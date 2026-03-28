## Context

### 范围

本 design 只覆盖 tile fusion 5.5 `PTOFusionRegionGenPass` 及其对应的 PTO dialect IR 容器定义。

前置条件固定为：

- 5.3 `FusionPlanPass` 已经完成分组并写出 `pto.fusion.group_id` / `pto.fusion.order`
- 5.4 `OpSchedulingPass` 已经让同组成员在单个 basic block 内形成连续片段
- 输入仍处于 `tile_buf world`

后置条件固定为：

- 每个合法 fusion group 被一个且仅一个 `pto.fusion_region` 封装
- region 输入、输出与 `yield` 顺序显式且稳定
- region 内不存在未声明的外部 SSA capture

### 当前状态

当前仓库里的相关实现仍然是 `PTOOutlineFusionGroups`，它会：

- 扫描连续组成员
- 生成 `@__pto_fused_group_*` helper function
- 在 caller 中插入 `func.call`
- 删除原 block 中的 group op

这条路径可以表达“把一段代码提取出去”，但它不等价于 5.5 设计文档中的 `pto.fusion_region`。helper function 会把融合边界编码成函数 ABI，而不是保留在原函数里的结构化 region。

### 约束

- 5.5 只在单个 basic block 内工作
- 不跨 region
- 不重新分组
- 不重新调度
- 不承担 5.6+ 的 memory / oplib / inline 消费者迁移
- 需要沿用 PTO dialect 现有 region-op 风格，而不是额外引入一套非 PTO 中间态约定

## Goals / Non-Goals

**Goals:**

- 定义 `pto.fusion_region` 作为 5.5 的正式结构化容器
- 定义 `pto.yield` 作为该容器的显式 terminator
- 实现 `PTOFusionRegionGenPass`，把已连续化的 fusion span 封装成 region
- 明确 region input / output / group identity / 闭包合法性契约
- 把 tile fusion 主线的 5.5 输出从 helper function 切换为 region

**Non-Goals:**

- 不修改 5.3 planning 的分组算法
- 不修改 5.4 scheduling 的合法性边界与移动策略
- 不在本阶段为 `PTOOutlineFusionGroups` 继续扩展 helper ABI
- 不在本阶段实现 5.6+ 对 `pto.fusion_region` 的消费逻辑
- 不做 CFG 变换

## Decisions

### 决策 1：5.5 直接以 region-based 输出替换 helper-outline 语义

5.5 的正式契约固定为：

- 成功封装后输出 `pto.fusion_region`
- 不再把 `@__pto_fused_group_*` helper function 作为 tile fusion 主线的目标形态

采用该方案的原因：

- 这与 `docs/tile_fusion/tile_fusion_design_spec.md` 中的 5.5 定义一致。
- 它把 tile fusion 的结构化边界保留在原函数中，便于后续 5.6+ pass 继续在局部范围内工作。

备选方案是继续复用 `PTOOutlineFusionGroups` 并把 helper function 视为 5.5。未采用该方案，因为它会把 region 边界退化成函数 ABI 约定，持续模糊 5.5 与后续阶段的职责分界。

### 决策 2：引入 `pto.fusion_region` 与 `pto.yield`，不复用 `scf.yield`

`pto.fusion_region` 和 `pto.yield` 都定义在 PTO dialect 中：

- `pto.fusion_region` 承载单 block body
- `pto.yield` 只允许作为其 body terminator

采用该方案的原因：

- 5.5 是 PTO tile fusion 专用中间态，使用 PTO dialect terminator 能把阶段语义和 verifier 约束固定在 PTO 体系内。
- 仓库里已经存在 `pto.section.*`、`pto.simd.vec_scope` 这类 PTO region op，风格连续。
- 后续 consumer 只需匹配 PTO dialect，不需要额外依赖“PTO 容器 + SCF terminator”的混合语义。

备选方案是定义 `pto.fusion_region` 但复用 `scf.yield`。未采用该方案，因为它会削弱 5.5 的专属 verifier 与阶段边界表达。

### 决策 3：`PTOFusionRegionGenPass` 只消费已经连续化的 span，不做补救性重排

`PTOFusionRegionGenPass` 的输入契约固定为：

- 同一 `pto.fusion.group_id` 在单个 basic block 中已经形成一个连续片段
- 片段内物理顺序已经满足 `pto.fusion.order`

采用该方案的原因：

- 这保持了 5.4 scheduling 与 5.5 encapsulation 的严格阶段边界。
- 若 5.5 发现 group 分裂或 metadata 残缺，应视为上游输入非法，而不是在本阶段隐式修复。

备选方案是在 5.5 中对不连续 group 做二次重排或 fallback。未采用该方案，因为这会把 5.4 的职责重新拉回 5.5。

### 决策 4：region interface 采用显式 external inputs / escaping outputs

封装算法分两步提取 interface：

- external inputs：group 内使用、但并非由 group 内定义的 SSA values
- escaping outputs：group 内定义、且在 group 外仍被使用的 SSA values

补充说明：

- “escaping” 的判定严格基于 span 外 SSA use，而不是“该值是组内最后一个写出的 destination”。
- 因此，若一个 terminal destination 在封装后不再被 parent block 中的任何 op 使用，`pto.fusion_region` 可以合法地产生空 result，并以空 `pto.yield` 结束。

顺序规则固定为：

- inputs 按“首次被组内 op 使用的程序顺序”稳定去重
- outputs 按“producer 程序顺序，再按结果位次”稳定排序

采用该方案的原因：

- 这使 region ABI 可预测、可测试，也便于 printer、verifier 和后续重写逻辑复现相同顺序。
- 它与 helper ABI 的“参数由外部传入”不同，但保留了显式边界的工程可读性。

备选方案是依赖隐式 capture 或只把 destination tiles 暴露为结果。未采用该方案，因为它会削弱 region 闭包性，并把边界语义隐藏到 op 细节里。

### 决策 5：成功封装后，per-op `pto.fusion.*` metadata 不再作为 5.5 后契约

成功生成 `pto.fusion_region` 后：

- `pto.fusion.group_id` 收敛到容器 op 上
- 原组内成员上的 `pto.fusion.group_id` / `pto.fusion.order` 作为 planning/scheduling 中间 metadata 被移除

采用该方案的原因：

- 5.5 之后，下游应消费结构化 region，而不是继续扫描零散 op 上的旧 metadata。
- 这能避免“region 已生成，但旧 attrs 仍被误当成正式输入”的双重契约。

备选方案是保留原 attrs 直到后续更晚阶段。未采用该方案，因为它容易让下游继续依赖旧路径，延迟完成阶段迁移。

## Risks / Trade-offs

- [Risk] 新增 PTO dialect region op 会增加 IR verifier、parser/printer 和 pass 接线工作量。
  → Mitigation：沿用现有 PTO region op 模式，只引入 `pto.fusion_region` 与 `pto.yield` 这组最小必要接口。

- [Risk] 若 input/output 提取顺序不稳定，测试与后续 consumer 会出现 nondeterministic 行为。
  → Mitigation：在 design 中固定顺序规则，并用 focused lit 覆盖多输入、多输出场景。

- [Risk] 移除原组内 attrs 可能影响尚未迁移的旧 helper-based consumer。
  → Mitigation：本 change 明确 tile fusion 主线已切到 region 契约；若仓库中仍保留旧 pass，只允许它作为独立旧路径存在，不能再被 5.5 正式链路依赖。

- [Risk] 若 5.4 输入不满足“单 group 对应单连续 span”，5.5 会频繁失败。
  → Mitigation：把失败定义成显式错误，并用 negative 回归锁定输入契约，问题回到 5.4 或更前阶段修复。

## Migration Plan

1. 在 PTO dialect 中新增 `pto.fusion_region` 和 `pto.yield`。
2. 声明并实现 `PTOFusionRegionGenPass`，直接消费 `pto.fusion.group_id` / `pto.fusion.order`。
3. 在 tile fusion pipeline 中把该 pass 放到 `OpSchedulingPass` 之后。
4. 让 5.5 正式链路停止以 `PTOOutlineFusionGroups` 作为目标输出契约。
5. 用 focused `lit` 回归和 `online_update` driver sample 验证 region 封装结果。

## Open Questions

- 本 change 不再保留设计层面的核心开放问题；默认决策已经固定为 `pto.fusion_region + pto.yield`，且 5.5 只负责封装不负责消费。
