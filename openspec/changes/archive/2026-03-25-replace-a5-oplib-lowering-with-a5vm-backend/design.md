## Context

### 范围

本 design 只覆盖 `tools/ptoas/ptoas.cpp` 中 A5 backend 主线的 pass pipeline 重整，以及与其直接相连的 tile fusion region handoff 边界。

本 change 覆盖的核心阶段为：

- `FusionPlanPass`
- `OpSchedulingPass`
- `PTOFusionRegionGenPass`
- `PTOToA5VM`
- `PTOLowLevelLoopFusion`
- `PTOFlattenFusionRegionPass`
- A5VM text / HIVM emission

本 change 不覆盖的内容：

- A3 主线的 backend 重构
- 所有 A5VM family 的 lowering 细节扩展
- 删除仓库中现有 OP-Lib 代码资产本身
- 对外新增新的 CLI 模式

### 当前状态

当前仓库里已经同时存在：

1. 旧的 A5 OP-Lib lowering 主线
   - `PTOInstantiateAndLowerToLibCall`
   - `PTOInlineLibCall`
   - 旧契约 `PTOLowLevelLoopFusion`
2. 新的 A5VM backend lowering 主线
   - `createLowerPTOToA5VMPass()`
   - A5VM text / LLVM/HIVM emitter

但 `tools/ptoas/ptoas.cpp` 仍然保留着以 OP-Lib 为中心的 A5 分流思路，导致：

- A5 backend 主线不明确，`EmitC`、旧 OP-Lib lowering 和 A5VM backend 三者交错。
- `PTOFusionRegionGenPass` 到 backend lowering 的 handoff 边界没有被正式锁定。
- `PTOLowLevelLoopFusion` 当前只识别旧 `pto.simd.vec_scope` / `vector.masked_*` 低层形态，不能直接满足“位于 `PTOToA5VM` 之后”这一目标顺序。

### 约束

- 必须保留 `PTOFusionRegionGenPass` 作为 tile fusion 5.5 结构化封装边界，不能回退到 helper-based outline 作为 A5 backend 正式输入。
- 必须让 `createLowerPTOToA5VMPass()` 成为 A5 backend 的正式 lowering 边界，而不是可选旁路。
- 必须保留 `EmitC` 输出路径，但不能再要求它穿过旧 OP-Lib lowering 主线。
- `PTOLowLevelLoopFusion` 的位置必须落在 `PTOToA5VM` 之后，因此实现契约必须随之改写，而不是只做 pipeline 顺序调整。

## Goals / Non-Goals

**Goals:**

- 把 `ptoas` A5 backend 主线固定为 `PTOFusionRegionGen -> PTOToA5VM -> PTOLowLevelLoopFusion -> PTOFlattenFusionRegion -> A5VM emit`。
- 移除 `ptoas.cpp` 中旧 OP-Lib lowering 作为 A5 backend 正式路径的角色。
- 让 `pto.fusion_region` 成为已计划 fusion group 在 `PTOToA5VM` 之前的稳定 handoff 边界，并在 A5VM post-lowering 阶段继续保持到显式 flatten。
- 重新定义 `PTOLowLevelLoopFusion` 的输入契约，使其服务于 A5VM lowering 之后的低层 loop 结构。
- 维持 `EmitC` 路径可用，但让它与旧 OP-Lib backend 解耦。

**Non-Goals:**

- 不在本 change 中归档或删除 `oplib-lowering` capability。
- 不在本 change 中把所有旧测试一次性迁移到 A5VM 语义。
- 不在本 change 中把 `PTOFlattenFusionRegionPass` 变成 legality 修复或重新调度 pass。
- 不在本 change 中引入新的 backend 选择参数或额外用户配置层级。

## Decisions

### 决策 1：A5 backend 的正式 lowering 边界固定为 `PTOToA5VM`

`ptoas.cpp` 中 A5 主线将不再以 `PTOInstantiateAndLowerToLibCall` / `PTOInlineLibCall` 为 backend 入口，而是改为以 `createLowerPTOToA5VMPass()` 为正式 backend lowering 边界。

采用该方案的原因：

- 当前仓库已经存在独立的 A5VM dialect、lowering pass 和 emitter，具备完整 backend 边界。
- 如果继续把 OP-Lib 链保留为 A5 主线，`PTOToA5VM` 会长期停留在“旁路 backend”状态，无法稳定演进。
- 这能把 “tile fusion 前半段” 和 “A5 backend lowering” 清晰分开。

备选方案是继续保留 OP-Lib 作为 A5 主线，并仅在个别场景切到 A5VM。未采用，因为这无法解决当前 `ptoas.cpp` backend 角色混杂的问题。

### 决策 2：`PTOFusionRegionGenPass` 固定前置到 `PTOToA5VM`

当启用 fusion 主线时，`PTOFusionRegionGenPass` 必须在 `PTOToA5VM` 之前完成，把 `pto.fusion_region` 作为进入 backend lowering 之前的正式结构化边界。

采用该方案的原因：

- `PTOFusionRegionGenPass` 已经是 5.5 的正式输出，后续 backend pass 应直接消费该结构化边界。
- 这能避免 backend lowering 再次回到“扫描散落 PTO op 并自行推断融合边界”的状态。
- 也能保证 `pto.yield` / region result frontier 在 backend lowering 前就已经稳定。

备选方案是让 `PTOToA5VM` 直接消费未封装的调度结果。未采用，因为这会把 5.5 封装责任重新拉回 backend pass，破坏阶段边界。

### 决策 3：`PTOLowLevelLoopFusion` 改为 A5VM post-lowering pass

`PTOLowLevelLoopFusion` 将保留在 `PTOToA5VM` 之后，但其输入契约必须从旧 `pto.simd.vec_scope` / `vector.masked_*` 形态迁移到 A5VM lowering 产出的低层 `scf.for + a5vm.*` 结构。

采用该方案的原因：

- 仅调整顺序而不调整契约会让该 pass 在新主线上长期 no-op。
- 用户已经明确要求它位于 `PTOToA5VM` 之后，因此实现必须跟随新的 IR 边界。
- 这让 low-level fusion 真正成为 A5VM backend 的 post-lowering 优化，而不是旧 OP-Lib/vector bridge 遗留逻辑。

备选方案一是把 `PTOLowLevelLoopFusion` 暂时留在 `PTOToA5VM` 之前。未采用，因为这与目标顺序冲突。  
备选方案二是把它放到 `PTOToA5VM` 之后但先接受 no-op。未采用，因为这会把关键优化退化成“位置正确但语义缺失”的假接入。

### 决策 4：planned fusion group 的 `pto.fusion_region` 在 A5VM post-lowering 期间继续保留，直到显式 flatten

对于已经计划并封装的 fusion group，`PTOToA5VM` 与新的 A5VM post-lowering `PTOLowLevelLoopFusion` 都在 `pto.fusion_region` body 内原位变换；`PTOFlattenFusionRegionPass` 仍然是唯一正式消解出口。与此同时，未被 fusion planning 接纳的 residual PTO op 继续允许保持在 region 外并由 `PTOToA5VM` 直接降低。

采用该方案的原因：

- 当前 `PTOToA5VM` 不负责消费 `pto.fusion_region` wrapper，本 design 不把 flatten 责任塞进 backend lowering。
- 保持 `pto.yield` / region result frontier 到 flatten 前，可避免在 A5VM post-lowering 期间丢失 region-local / escaping value 边界。
- 这与现有 tile-fusion-region-lowering capability 的结构化边界方向一致，只是把下游 pass 从 OP-Lib 改成 A5VM，同时不扩大 `fusion_region` 的语义范围。

备选方案一是让 `PTOToA5VM` 直接顺手 flatten region。未采用，因为这会把 backend lowering 和 region 生命周期管理耦合在一起。  
备选方案二是把 residual non-fused PTO op 也补包成 `pto.fusion_region` 后再降低。未采用，因为这会把 `fusion_region` 从“fusion group 边界”扩张成“通用 backend wrapper”，偏离 5.5 的计划语义。

### 决策 5：保留 `EmitC` 输出，但与旧 OP-Lib lowering 解耦

`EmitC` 路径继续保留为可选输出，但不再把旧 OP-Lib lowering 视作它的前置依赖，也不再让 `--op-lib-dir` 成为 A5 EmitC 的正式输入。

采用该方案的原因：

- 用户已经明确要求保留 EmitC 路径。
- 这次 change 的目标是清理 A5 backend 边界，而不是强制删除所有非 A5VM 输出形式。
- 将 EmitC 与 OP-Lib lowering 解耦后，`EmitC` 和 `A5VM` 才能作为两个清晰的最终输出分支存在。

备选方案是完全移除 A5 EmitC。未采用，因为不符合当前需求。

## Risks / Trade-offs

- [Risk] `PTOLowLevelLoopFusion` 从旧 vector bridge 迁到 A5VM 后，旧基于 `vector.masked_*` 的回归会失效。  
  → Mitigation：在 spec 中明确该 pass 的新责任边界，并分别补 A5VM IR 顺序与 post-lowering fusion regression。

- [Risk] `PTOToA5VM` 后继续保留 planned fusion group 的 `pto.fusion_region`，若后续 pass 未遵守 frontier 约束，可能绕过 `pto.yield`。  
  → Mitigation：继续把 flatten 保持为单独 pass，并在 spec 中要求 backend/post-lowering pass 对已封装 fusion group 只在 region body 内原位变换。

- [Risk] `EmitC` 与旧 OP-Lib 解耦后，部分现有 A5 EmitC 样例可能暴露出对旧 pipeline 的隐藏依赖。  
  → Mitigation：把本 change 的 contract 收口在 backend 边界与 pass 顺序，要求实现补 focused smoke regression，而不是承诺旧样例零改动。

- [Risk] 旧 CLI 语义，尤其是 `--op-lib-dir`，在本 change 后会变成历史兼容项。  
  → Mitigation：设计上允许保留参数入口，但不再把它作为 A5 主线 backend 的正式配置源。

## Migration Plan

1. 先在 OpenSpec 中固定新的 A5 backend 主线、`fusion_region` handoff 边界和 A5VM post-lowering loop fusion 契约。
2. 重整 `tools/ptoas/ptoas.cpp`：
   - 删除旧 OP-Lib backend 主线分支
   - 重排 `PTOFusionRegionGenPass` / `PTOToA5VM` / `PTOLowLevelLoopFusion`
   - 保留 `EmitC` 但移除其对旧 OP-Lib 主线的依赖
3. 改写 `PTOLowLevelLoopFusion` 的输入契约和实现，使其直接匹配 A5VM lowering 后的低层 loop 结构。
4. 更新 A5VM backend 与 tile fusion focused regression，锁定 pass 顺序和 region 生命周期。

回滚策略：

- 若 A5VM post-lowering `PTOLowLevelLoopFusion` 在实现期未稳定，可先回滚该 pass 的功能改写，但不回滚 OpenSpec 中对 A5 backend 主线的整理方向。
- 若 `EmitC` 与旧 OP-Lib 解耦暴露出大量兼容问题，可暂时在实现中保留兼容 warning/fallback，但不恢复 OP-Lib 作为 A5 主线 backend 的正式地位。

## Open Questions

本 change 不保留设计级开放问题。默认决策已经固定为：

- `PTOFusionRegionGenPass` 在 `PTOToA5VM` 之前
- `PTOLowLevelLoopFusion` 在 `PTOToA5VM` 之后
- A5 正式 backend lowering 边界为 `PTOToA5VM`
- `EmitC` 保留，但不再依赖旧 OP-Lib lowering 主线
