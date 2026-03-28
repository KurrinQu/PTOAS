## Context

### 范围

本 design 只覆盖 tile fusion 5.5 `pto.fusion_region` 之后、进入 Emit/手工降级之前的过渡主线，目标是把 tile fusion 正式推进到 `LowerToLibCall`。

本 change 覆盖的阶段边界为：

- `PTOFusionRegionGenPass` 之后的 `pto.fusion_region`
- 当前 memref-world `PlanMemory`
- 当前 memref-world `PTOInsertSync`
- `PTOInstantiateAndLowerToLibCall`
- `PTOInlineLibCall`
- `PTOLowLevelLoopFusion`
- 新增的显式 flatten 出口

本 change 内的首批新增 lowering scope 固定为：

- 当前 region-based tile fusion 热点链路所需的 `trowexpandmul`

不在本 design 范围内的内容：

- 把 `PlanMemory` / `InsertSync` 迁回 tile_buf world
- 回退到 helper-based `PTOOutlineFusionGroups` 主线
- 引入 chain-level monolithic OP-Lib template
- 一次性扩展所有 prelude / broadcast / reduction / transpose family

### 当前状态

当前 `ptoas` 主线在 A5 + fusion 打开时，实际 pass 顺序仍以以下结构为准：

1. `FusionPlanPass -> OpSchedulingPass -> PTOFusionRegionGenPass`
2. `PTOViewToMemref`
3. `PlanMemory -> PTOInsertSync -> PTOInstantiateAndLowerToLibCall -> PTOInlineLibCall -> PTOLowLevelLoopFusion`

这意味着：

- 5.5 已经切到 `pto.fusion_region` 正式输出；
- 但 `PlanMemory` / `PTOInsertSync` 仍建立在 memref-world、block-local 线性 op 建模之上；
- grouped `PTOInstantiateAndLowerToLibCall` 仍主要按“普通函数体 / helper 内 op 序列”工作，而不是把 `pto.fusion_region` 当成一等 lowering unit。

当前仓库里的实际阻塞已经可以复现为两个具体现象：

1. `test/tile_fusion/fusion_region_interface.mlir` 会在 `PlanMemory` 报出 `PlanMemory Fail : Unrecognized type of Operation touches local buffer!`，说明 `pto.fusion_region` 目前不能穿过现有 memref-world `PlanMemory`。
2. `test/tile_fusion/fusion_region_basic.mlir` 能在 region 内继续完成 `PTOInstantiateAndLowerToLibCall` / `PTOInlineLibCall`，说明当前 `PTOLowerToOpLibCalls.cpp` 的 tile-like ABI 兼容能力可以被复用；但最终仍会因为 residual `pto.fusion_region` 缺少正式消解路径而在 Emit/手工降级前失败。

### 约束

- 必须保留 5.5 `pto.fusion_region` 作为 tile fusion 主线的正式输出契约，不能为了这一步推进而重新回退到 helper ABI。
- 必须优先复用当前 `PTOLowerToOpLibCalls.cpp` 中已有的 memref-world template selection / instantiation / `SimdTileToMemrefOp` remap 能力，避免重复造一套新的 grouped lowering 框架。
- 必须尽量保持 `PlanMemory` / `InsertSync` 的既有核心逻辑不变；本 change 只要求它们对 `pto.fusion_region` 透明，而不是一次性变成“懂 fusion 优化”的新分析器。
- 实现必须为未来 tile_buf-world 最终版留出平滑迁移路径，避免把 memref-world 过渡方案固化成新的长期耦合。

## Goals / Non-Goals

**Goals:**

- 让 `pto.fusion_region` 在当前 memref-world `PlanMemory` / `PTOInsertSync` 中稳定存在。
- 让 grouped `PTOInstantiateAndLowerToLibCall` 直接消费 `pto.fusion_region` 作为 lowering unit。
- 把 `trowexpandmul` 纳入 tile fusion region path 的首批 active grouped OpLib lowering scope。
- 让 `PTOInlineLibCall` / `PTOLowLevelLoopFusion` 继续在 region body 内工作，并通过显式 flatten 在 Emit 前消解 region。
- 使这一步的结构与未来最终版兼容：将来只需把 `PlanMemory` / `InsertSync` 迁回 tile_buf world，并把 LowerToLibCall 的 world adapter 切回 tile_buf ABI。

**Non-Goals:**

- 不把 `LowerToLibCall` 立即前移到 `PlanMemory` 之前。
- 不要求 `PlanMemory` / `PTOInsertSync` 在本 change 中理解 OP-Lib call / inline 后 `vec_scope` 级融合语义。
- 不扩展所有 row-broadcast / broadcast / reduction family。
- 不让 `PTOFlattenFusionRegionPass` 承担重排、再分组或 legality 修复职责。

## Decisions

### 决策 1：保持当前大阶段顺序，不前移 `LowerToLibCall`

本 change 固定采用：

- `PTOFusionRegionGenPass`
- `PTOViewToMemref`
- `PlanMemory`
- `PTOInsertSync`
- `PTOInstantiateAndLowerToLibCall`
- `PTOInlineLibCall`
- `PTOLowLevelLoopFusion`
- `PTOFlattenFusionRegionPass`

而不是把 `LowerToLibCall` 前移到 `PlanMemory` 之前。

采用该方案的原因：

- 这条路径与最终目标“`InsertSync` 之后再 `LowerToLibCall`”的阶段边界更接近。
- 当前 `PTOLowerToOpLibCalls.cpp` 已经具备 memref-world tile-like ABI 兼容能力，复用成本低。
- 反过来让 `PlanMemory` / `InsertSync` 直接面对 OP-Lib call、`pto.simd.tile_to_memref`、`pto.simd.vec_scope` 和 inline 后 loop nest，会显著扩大过渡实现的改造面。

备选方案是立即把 `LowerToLibCall` 前移到 `PlanMemory` 前的 tile_buf world。未采用该方案，因为它会把本应后续完成的 tile_buf-world `PlanMemory` / `InsertSync` 迁移和 OP-Lib 适配绑在同一 change 中，风险过高。

### 决策 2：让 `PlanMemory` / `PTOInsertSync` 对 `pto.fusion_region` 透明，而不是在 wrapper 上建新语义

本 change 中：

- `PlanMemory` 必须递归进入 `pto.fusion_region` body；
- `pto.yield` operands 与 `pto.fusion_region` results 之间必须建立 alias / frontier 关系；
- wrapper 本身不产生独立的 memory-touching compute 语义；
- `PTOInsertSync` 也必须递归进入 body，并把 `pto.yield` 当作外部可见边界，而不是在 wrapper 本身生成 sync node。

采用该方案的原因：

- 这能最大限度保留 `PlanMemory` / `PTOInsertSync` 的现有分析与代码生成逻辑。
- 它把 `pto.fusion_region` 明确成“局部结构化容器”，而不是新增一层需要专门建模的 compute barrier。
- 将来切回 tile_buf world 时，这一层“透明容器 + frontier alias”的支持仍可复用。

备选方案是在 `PlanMemory` / `PTOInsertSync` 上直接为 `pto.fusion_region` 定义新的一级 memory/sync 语义。未采用该方案，因为它会让过渡 change 过早承担完整 5.6+ 语义设计。

### 决策 3：grouped `PTOInstantiateAndLowerToLibCall` 直接消费 `pto.fusion_region`

本 change 里，grouped lowering 的正式输入单元改为 `pto.fusion_region`，而不是继续扫描 5.5 前的 per-op `pto.fusion.group_id` / `pto.fusion.order`。

具体约束为：

- 一个 `pto.fusion_region` 对应一个 grouped lowering unit。
- region body 内允许存在 local alloc / bind / pointer_cast / view 一类桥接 op。
- region body 内的 compute op 必须全部属于当前 active region lowering scope，否则 pass 必须 hard-fail。

采用该方案的原因：

- 5.5 成功后，原组内成员上的 `pto.fusion.*` metadata 已经被移除，继续依赖旧扫描逻辑并不稳固。
- region 本身已经是 5.5 之后唯一合法的结构化边界，下游 pass 应直接消费这个边界。
- 当前 `PTOLowerToOpLibCalls.cpp` 已能在 region body 中重写 compute op，说明这一迁移是增量的，而不是全新通路。

备选方案是重新引入 helper outline 作为 grouped lowering unit，或继续在 region body 内依赖旧 attrs。未采用，因为这会重新模糊 5.5 之后的正式输入契约。

### 决策 4：LowerToLibCall 内部按 “planner + world adapter” 拆分，当前先落 memref-world adapter

实现上，`PTOLowerToOpLibCalls.cpp` 需要拆成两层：

1. planner 层：
   - 扫描 active compute op
   - 基于 `OpLibOpInterface` / `OpLibMatchDescriptor` 选择 template / variant
   - 建立 grouped lowering plan
2. world adapter 层：
   - 负责 tile_buf 或 memref tile-like operand 的具体物化
   - 当前 change 先落 memref-world adapter

采用该方案的原因：

- 这能把“模板选择语义”与“当前所处 world 的 ABI 物化方式”拆开。
- 当前 change 可以直接复用 memref-world path，后续最终版只需替换 adapter，并把 pass 位置后移到 tile_buf-world `InsertSync` 之后。
- 它避免把“过渡期实现细节”写死在 family / matcher 语义层。

备选方案是直接在当前 pass 里继续堆叠 region + memref 的 ad hoc 特判。未采用，因为那样会把未来 tile_buf-world 迁移成本留到更难处理的状态。

### 决策 5：首批 prelude scope 固定覆盖 `trowexpandmul`，并对 partially-supported region hard-fail

本 change 的 prelude lowering scope 不做无限扩展，而是收口到当前推进 tile fusion LowerToLibCall 所必需的第一波 op：

- `trowexpandmul`

同时固定以下行为：

- 若 region 内出现 `trowexpandmul` 与当前既有 arithmetic active scope 的混合链，只要全部属于 active scope，就必须完整 LowerToLibCall。
- 若 region 内出现 active op 与 unsupported compute op 混合，则 pass 必须 deterministic hard-fail，而不是部分 lower。

采用该方案的原因：

- `trowexpandmul` 是当前热点样例中实际阻塞 LowerToLibCall 推进的 prelude op。
- 先把真实热点打通，比在一个 change 里承诺整个 prelude / broadcast 家族更稳妥。
- partially-lowered region 会让后续 inline / low-level fusion / flatten 的中间态更难定义，hard-fail 更安全。

备选方案一是把 scope 保持在现有 12 个 elementwise op。未采用，因为这无法覆盖当前要推进的 region 前导链路。  
备选方案二是一次性纳入全部 row-broadcast / broadcast family。未采用，因为这会让过渡 change 再次失焦。

### 决策 6：新增 `PTOFlattenFusionRegionPass` 作为唯一正式 region 消解出口

`pto.fusion_region` 在 5.5 之后到 low-level fusion 结束之前继续作为结构化边界存在；最终由新增的 `PTOFlattenFusionRegionPass` 统一消解：

- splice body 回父 block
- 用 `pto.yield` operands 替换 region results
- 删除 `pto.yield` 与 `pto.fusion_region`

采用该方案的原因：

- 它让 `PTOInlineLibCall` / `PTOLowLevelLoopFusion` 不需要兼做 wrapper legalize。
- 它把“region 生命周期结束点”固定在单独 pass 上，便于测试和回滚。
- 它避免把 Emit/手工降级被动变成 `pto.fusion_region` 的 fallback consumer。

备选方案是让 Emit/手工降级直接支持 residual `pto.fusion_region`。未采用，因为那会把一个过渡期结构化边界长期泄漏到更后段。

## Risks / Trade-offs

- [Risk] memref-world 过渡方案可能形成新的长期耦合。  
  → Mitigation：在 design 上明确 planner / world adapter 分层，并把最终 tile_buf-world 迁移保留为替换 adapter 与 pass 位置，而不是重做 matcher 语义。

- [Risk] 让 `PlanMemory` / `PTOInsertSync` 递归进入 `pto.fusion_region` 可能暴露新的 alias/frontier 边界 bug。  
  → Mitigation：把 `pto.yield` operand -> region result alias 规则写成明确契约，并增加 focused regression 覆盖 escaping result 与空 result 两类 region。

- [Risk] `trowexpandmul` 加入 active grouped scope 后，若 row-broadcast 角色在 matcher 中丢失，会选错 family / variant。  
  → Mitigation：要求 grouped path 继续走 `OpLibOpInterface` / `OpLibMatchDescriptor`，并在 spec 与测试中显式校验 full-tile vs row-broadcast 角色保留。

- [Risk] partially-supported region hard-fail 会让部分当前样例在 change 中期暂时失效。  
  → Mitigation：在实现顺序上先补齐 region-transparent `PlanMemory` / `InsertSync` 与 `trowexpandmul` lowering，再把 region-based lowering path 挂到默认 pipeline。

- [Risk] flatten pass 若顺序放错，可能破坏 low-level fusion 看见的 canonical region-internal `vec_scope` 形态。  
  → Mitigation：明确 flatten 只能放在 `PTOLowLevelLoopFusion` 之后、Emit/手工降级之前，并用 IR regression 锁定顺序。

## Migration Plan

1. 先补 proposal / design / specs，固定 `pto.fusion_region` 下游消费契约与 `oplib-lowering` 范围。
2. 修改 `PlanMemory` 与 `PTOInsertSync` 的递归/alias 路径，使 `pto.fusion_region` 对现有分析透明。
3. 调整 `PTOLowerToOpLibCalls.cpp`，把 grouped lowering 输入切到 `pto.fusion_region`，并补齐 `trowexpandmul`。
4. 新增 `PTOFlattenFusionRegionPass`，接到 `PTOLowLevelLoopFusion` 之后。
5. 用 `test/tile_fusion/` 与 `test/oplib/` focused regression 锁定：
   - region-transparent `PlanMemory` / `InsertSync`
   - region-based LowerToLibCall
   - partially-supported region hard-fail
   - flatten 后无 residual region

回滚策略：

- 若 `PlanMemory` / `PTOInsertSync` 的 region-transparent 改造未稳定，先不把 region-based LowerToLibCall 和 flatten pass 接入默认 pipeline。
- 若 region-based LowerToLibCall 稳定但 flatten 仍有问题，保留 region-internal lowering 代码，不把 flatten 挂入默认链路，避免把 residual region 带到 Emit。

## Open Questions

本 change 不保留核心设计级开放问题。默认决策已经固定为：

- 不前移 `LowerToLibCall`
- `PlanMemory` / `PTOInsertSync` 对 `pto.fusion_region` 透明
- grouped lowering 直接消费 `pto.fusion_region`
- 首批 prelude scope 固定为 `trowexpandmul`
- flatten 作为唯一正式 region 消解出口
