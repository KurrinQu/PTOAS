# tile-fusion-post-lowering-unroll Specification

## Purpose
TBD - created by archiving change add-a5vm-post-fusion-loop-unroll. Update Purpose after archive.
## Requirements
### Requirement: Post-fusion loop unroll MUST stay inside `pto.fusion_region`

`PTOPostFusionLoopUnroll` MUST 只作用于 `pto.fusion_region` body 内的 post-cleanup carrier loop，不得把该阶段扩张到 wrapper 外的 residual non-fused A5VM op。

#### Scenario: Residual non-fused A5VM ops remain untouched

- **WHEN** 某个函数同时包含 `pto.fusion_region` body 与该 region 外部的 residual non-fused A5VM op
- **THEN** `PTOPostFusionLoopUnroll` MUST 只处理 `pto.fusion_region` body 内的候选 loop
- **AND** MUST 保持 parent block 中非 fusion A5VM op 不变

### Requirement: Unroll cost model MUST consume post-fusion post-cleanup loop bodies

`PTOPostFusionLoopUnroll` 的正式输入 MUST 是已经完成 `PTOLowLevelLoopFusion`、`Canonicalizer/CSE`、`PTOFusionPredicateElision` 和 `PTOFusionLoadStoreElision` 的 carrier loop。  
它 MUST 以这些 post-cleanup 低层 loop 为依据做决策，不得回退到 planner/pre-lowering 阶段的 tile-level 估计结果作为正式输入。

#### Scenario: Cost model sees cleaned carrier loop rather than planner-time estimate

- **WHEN** `PTOPostFusionLoopUnroll` 对某个 fusion-local loop 做是否展开的决策
- **THEN** 它 MUST 基于 post-fusion post-cleanup 的 `scf.for + a5vm.*` body 判断收益与风险
- **AND** MUST NOT 仅根据 planner、grouping 或 pre-lowering 阶段记录的抽象 chain 信息做最终决策

### Requirement: Unroll decisions MUST remain conservative and factor-limited

`v1` contract 下，`PTOPostFusionLoopUnroll` MUST 只允许 `skip`、`unroll x2`、`unroll x4` 这类小而稳的决策集合。  
该阶段的 cost model MUST 至少综合考虑 loop body 指令数、trip count / valid shape 可证性、live value / loop-carried value 粗估、tail 比例、predicate 复杂度和粗粒度寄存器压力。  
当收益不可证、成本过高、控制流过于复杂或寄存器压力风险过大时，transform MUST 保守 no-op。

#### Scenario: Small factor set is used for profitable steady-state loop

- **WHEN** 某个 fusion-local carrier loop 的 body 足够短、trip count 合理、tail 开销可控且寄存器压力风险可接受
- **THEN** `PTOPostFusionLoopUnroll` MAY 选择 `unroll x2` 或 `unroll x4`
- **AND** MUST NOT 在 `v1` 中直接选择超出该小因子集合的更激进展开

#### Scenario: High-cost candidate remains unchanged

- **WHEN** 某个候选 loop 的 tail 比例过高、loop-carried value 过多、predicate 复杂度过大或寄存器压力风险不可接受
- **THEN** `PTOPostFusionLoopUnroll` MUST 保守保持原 loop 不变
- **AND** MUST NOT 为追求展开而牺牲当前主线的稳定性

### Requirement: Dynamic valid shape MUST preserve explicit tail semantics

`PTOPostFusionLoopUnroll` MUST 同时支持静态 shape 与动态 valid shape。  
当 trip count 可在编译期证明且整除展开因子时，transform MAY 做无额外 tail 的部分 unroll；当 trip count 很小且完全可证时，MAY 退化为小规模 full unroll。  
对动态 valid shape，transform MAY 做部分 unroll，但 MUST 保留等价的显式 tail 处理，通常体现为 residual loop 或等价 predicate 路径；MUST NOT 依赖“运行时恰好整除”的隐式假设。

#### Scenario: Static divisible loop unrolls without extra tail

- **WHEN** 某个候选 loop 的 trip count 在编译期可证且能被选定因子整除
- **THEN** `PTOPostFusionLoopUnroll` MAY 生成无额外 tail 的部分 unroll 形态

#### Scenario: Dynamic valid shape keeps explicit tail handling

- **WHEN** 某个候选 loop 的 valid shape 或 trip count 含有动态分量
- **THEN** `PTOPostFusionLoopUnroll` MAY 做部分 unroll
- **AND** MUST 保留等价的显式 tail 处理
- **AND** MUST NOT 仅依赖“动态 trip count 恰好整除”来省略 tail

### Requirement: Unroll MUST NOT reintroduce fusion-local round-trip memory traffic

展开后的 loop body MUST 保持 `PTOFusionLoadStoreElision` 已建立的 fusion-local 数据通路收缩结果，不得为了展开重新引入仅供链内消费的 `a5vm.vsts -> a5vm.vlds` round-trip。

#### Scenario: Expanded loop still forwards fusion-local values without round-trip

- **WHEN** `PTOPostFusionLoopUnroll` 对某个已经过 load/store cleanup 的 carrier loop 做展开
- **THEN** 展开后的 loop body MUST 继续复用 fusion-local SSA/value 通路
- **AND** MUST NOT 为了复制 stage body 而重新物化仅链内可见的中间 store/load 往返

### Requirement: Unroll MUST preserve fusion-region frontier and flatten compatibility

`PTOPostFusionLoopUnroll` MUST 保持 `pto.yield` / `pto.fusion_region` result 作为 region 对外可见值的正式 frontier。  
展开过程中 MAY 重写 region body 内部 loop，但 MUST NOT 绕过或改写该正式边界，并且 MUST 保证 `PTOFlattenFusionRegion` 后续仍可按既有契约直接 splice body。

#### Scenario: Flatten remains valid after unroll

- **WHEN** `PTOPostFusionLoopUnroll` 已对某个 `pto.fusion_region` body 内的 carrier loop 做了展开
- **THEN** `pto.yield` / region result frontier MUST 继续保持不变
- **AND** `PTOFlattenFusionRegion` MUST 仍能直接展开该 region，不需要新增额外 wrapper 或 synthetic frontier

