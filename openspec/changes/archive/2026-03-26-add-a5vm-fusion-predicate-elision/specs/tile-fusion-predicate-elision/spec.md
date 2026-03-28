## ADDED Requirements

### Requirement: Fusion-region predicate elision MUST stay inside `pto.fusion_region`

冗余 `plt` 消除 MUST 只作用于 `pto.fusion_region` body 内的 A5VM IR，不得把新契约扩张到 wrapper 外的 residual non-fused A5VM op。

#### Scenario: Residual non-fused A5VM ops remain untouched

- **WHEN** 某个函数同时包含 `pto.fusion_region` body 与该 region 外部的 residual non-fused A5VM op
- **THEN** predicate-elision MUST 只改写 `pto.fusion_region` body 内的候选 `plt`
- **AND** MUST 保持 parent block 中非 fusion A5VM op 不变

### Requirement: Equivalent fusion-local `plt` ops MUST be deduplicated as whole-result pairs

对同一 `pto.fusion_region` 内、由前一个 op 支配后一个 op 的等价 `a5vm.plt_b8`、`a5vm.plt_b16`、`a5vm.plt_b32`，transform MUST 复用前一个 op 的 `mask` 与 `scalar_out` 两个结果，并删除后一个冗余 op。

#### Scenario: Dominating equivalent `plt_b32` is reused for both results

- **WHEN** 一个 fusion region body 中存在两个 `a5vm.plt_b32`
- **AND** 前一个 `plt_b32` 支配后一个 `plt_b32`
- **AND** 两者的 scalar 输入被证明等价
- **THEN** predicate-elision MUST 用前一个 `plt_b32` 的 `mask` 与 `scalar_out` 替换后一个 `plt_b32` 的全部使用
- **AND** MUST 删除后一个冗余 `plt_b32`

#### Scenario: Same rule applies to `plt_b8` and `plt_b16`

- **WHEN** `a5vm.plt_b8` 或 `a5vm.plt_b16` 在同一 fusion region 内满足相同的支配与输入等价条件
- **THEN** predicate-elision MUST 以与 `plt_b32` 相同的整 op 双结果复用规则处理它们

### Requirement: Loop-carried scalar equality MUST participate in `plt` dedup inside supported recurrences

对于 low-level fusion 后支持的 `scf.for` recurrence，若多个 iter_arg 由等价 init 值出发，并持续由等价 `plt.scalar_out` 递推，则 predicate-elision MUST 将这些 loop-carried scalar 视为等价输入源，用于识别后续冗余 `plt`。

#### Scenario: Equivalent iter_args collapse duplicate `plt` materialization

- **WHEN** 某个 `scf.for` 的两个 iter_arg 具有等价 init 值
- **AND** 上一轮 `scf.yield` 中这两个 iter_arg 的更新值都来自等价 `plt.scalar_out`
- **AND** 下一轮迭代内又分别对这两个 iter_arg 执行同一 bitwidth 的 `a5vm.plt_*`
- **THEN** predicate-elision MUST 将这两个 iter_arg 视为等价 scalar 输入
- **AND** MUST 复用前一个支配 `plt` 的 `mask` 与 `scalar_out`
- **AND** MUST NOT 仅因为两个 iter_arg 不是同一个 SSA value 就放弃消除

### Requirement: Predicate elision MUST remain conservative when equivalence is not provable

当 `plt` 的 bitwidth 不同、支配关系不成立、loop-carried recurrence 不再等价，或当前 IR 形态超出支持的证明域时，predicate-elision MUST 保守保持原状。

#### Scenario: Different bitwidth or divergent recurrence keeps both `plt`

- **WHEN** 两个候选 `plt` 的 op kind 不同、scalar 输入不等价，或对应 iter_arg recurrence 已经分叉
- **THEN** predicate-elision MUST 保留两个 `plt`
- **AND** MUST NOT 生成跨 bitwidth、跨 recurrence 或无支配关系的结果复用
