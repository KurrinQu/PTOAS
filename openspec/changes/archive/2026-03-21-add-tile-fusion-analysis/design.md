## Context

### 范围

本 design 只覆盖 `docs/tile_fusion/tile_fusion_design_spec.md` 中的 5.1 `PreFusionAnalysisPass`。
它不覆盖 5.3 的分组/planning，也不覆盖 5.4 的调度/scheduling。

本 change 的边界固定为：

- 工作在 `tile_buf world`
- 同时支持 SSA / DPS 两种输入形态
- 不要求强依赖 `PTOConvertToDPS`，因此可放在其前或后，只要仍处于 `tile_buf world`
- 不做分组、不做调度、不做 5.5+ 后段变换
- 5.2 暂不实现，动态符号等价不可证时一律保守

驱动样例仍是：

- `test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto`

### 当前状态

当前仓库没有独立的 tile fusion 分析层。现有 `PTOCreateFusionGroups` 在做的事情更接近一个“线性链分组 pass”，它缺少：

- DFG
- 生命周期分析
- 迭代域等价类
- SSA / DPS 双输入抽象
- 对 `treshape` 的稳定语义建模

这使得后续 5.3 / 5.4 无法围绕一个共享的分析结果工作。

### 约束

- 只针对当前 planning 范围内的 compute family 建模：
  - 现有 12 个 binary / binary-scalar elementwise
  - `texp`
  - `texpands`
  - `trowexpandmul`
  - `trowexpanddiv`
- `treshape` 不属于 compute family。
- `tload` / `tstore` / barrier / side-effect / 外部 call / terminator 保持边界语义，不作为融合 compute 节点。

## Goals / Non-Goals

**Goals:**

- 建立统一的 `PreFusionAnalysisPass` 契约和分析结果类型。
- 将 SSA / DPS 两种输入归一到同一 producer / consumer 语义。
- 为后续阶段输出 DFG、生命周期和迭代域信息。
- 明确 `treshape` 的局部非穿透边界语义。

**Non-Goals:**

- 不实现任何分组/planning 决策。
- 不实现任何调度/scheduling 决策。
- 不输出 `fusion_id` 或 `pto.fusion.order`。
- 不做任何 IR 重排。
- 不进行动态 shape 推导或符号统一化。
- 不提供 5.5+ 后段所需的 materialization 细节。

## Decisions

### 决策 1：用统一的 `FusionOpSemantics` 归一化 SSA / DPS 输入

`PreFusionAnalysisPass` 不直接在原始 IR 上拼接分支式逻辑，而是先把目标 op 规约到统一语义：

- `ComputeOp`
  - tile inputs
  - tile outputs
  - scalar inputs
  - op family
  - 迭代域描述
- `LocalBoundaryOp`
  - 当前只用于 `treshape`
- `HardBoundaryOp`
  - `tload` / `tstore` / barrier / side-effect / 外部 call / terminator

SSA / DPS 的统一方式：

- SSA：直接读取结果 value 与 operand 关系
- DPS：通过 `PTO_DpsInitOpInterface` 恢复 destination，再结合 op 接口恢复输入角色

采用该方案的原因：

- 后续 5.3 / 5.4 可以完全围绕统一语义层写逻辑。
- 不需要把 “SSA 专用路径” 和 “DPS 专用路径” 再拆成两套 planning 代码。

### 决策 2：`treshape` 作为局部非穿透边界，而不是 relay 或全局屏障

本 change 固定 `treshape` 的分析语义：

- `OPA -> treshape -> OPB` 这条链不建立可穿透的融合依赖
- `treshape` 不进入 compute DFG
- `treshape` 也不把与其无关的 op 整体隔离出当前 block-local 候选区域

换句话说：

- 它对“经过它的依赖链”是边界
- 它对“与它无关的其他 op”不是边界

采用该方案的原因：

- 这与用户明确给出的实现边界一致。
- 它避免把 `treshape` 错当成普通 compute，也避免把它升级成全局调度屏障。

### 决策 3：分析结果只对 block-local 图给出结论

为了和 5.3 / 5.4 的落地范围一致，`PreFusionAnalysisPass` 只输出 block-local 级别的图与分类信息：

- basic block 内建图
- region 内分别分析
- 不跨 block 合并 DFG
- 不跨 region / side-effect / barrier 建立融合候选边

采用该方案的原因：

- online_update 当前需要的是 block-local DAG，而不是跨 block 的全局图划分。
- 这能显著降低第一版分析层的复杂度。

### 决策 4：动态迭代域不可证时，analysis 显式输出“不可证”

`PreFusionAnalysisPass` 不尝试替代 5.2。对于需要动态 shape 等价证明的场景，analysis 结果必须显式带出“不可证”状态，而不是偷偷猜测为相等。

采用该方案的原因：

- 这样 5.3 可以直接把这类 case 拒绝，不必重复推断。
- 这符合“5.2 暂不实现”的阶段边界。

## Risks / Trade-offs

- [Risk] SSA / DPS 归一化若覆盖不完整，会让一部分目标 op 缺少依赖边。
  → Mitigation：v1 仅覆盖当前规划范围内的 family；其余 op 统一降级为边界。

- [Risk] 把 `treshape` 当作局部非穿透边界，会让某些本来潜在可融合的链路在 v1 被保守拒绝。
  → Mitigation：这正是当前阶段的显式选择；相关机会留给后续 change 重新评估。

- [Risk] 仅做 block-local 分析会限制未来跨 block 融合。
  → Mitigation：v1 明确不覆盖跨 block 融合，后续若需要再单独扩展。

## Migration Plan

1. 新增 `PreFusionAnalysisPass` 的声明、注册和分析结果类型。
2. 实现 SSA / DPS 统一语义归一化。
3. 实现 DFG、生命周期和迭代域分类输出。
4. 补充可观测的 analysis dump 或等价测试钩子。
5. 用 focused lit 覆盖 SSA / DPS / `treshape` / dynamic-shape negative case。

## Open Questions

- analysis dump 是通过显式 debug printer pass 暴露，还是通过 `PreFusionAnalysisPass` 的 debug logging 暴露，更利于回归维护？
- `texpands` 是否在 analysis 层直接当作完整 compute op，还是保留为一种带特殊 cost hint 的 seed compute，后续可再细化？
