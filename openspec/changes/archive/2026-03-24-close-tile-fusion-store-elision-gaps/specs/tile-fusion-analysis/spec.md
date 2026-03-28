# Tile Fusion Analysis Specification

## ADDED Requirements

### Requirement: PreFusionAnalysisPass MUST track distinct write instances for DPS-reused destinations

`PreFusionAnalysisPass` MUST 区分“tile storage”与“对该 storage 的一次逻辑写入”。当多个 compute op 复用同一个 DPS destination tile 时，analysis MUST 为每次写建立独立的 write instance，而不是只保留最后一次写覆盖前序 producer 记录。

#### Scenario: DPS destination reuse does not collapse multiple writes

- **WHEN** 同一 basic block 中多个目标 op 依次对同一个 DPS destination tile 执行写入
- **THEN** `PreFusionAnalysisPass` MUST 为每个写入生成独立的 producer / consumer / last-local-consumer 记录
- **AND** MUST 保留这些写实例与底层 tile storage 的关联关系
- **AND** MUST NOT 让后序写实例覆盖前序写实例的 analysis 结果

### Requirement: PreFusionAnalysisPass MUST classify tile live ranges by escape kind

除 DFG、iteration-domain 和普通 liveness 外，`PreFusionAnalysisPass` MUST 为每个 write instance 输出稳定的 escape classification，至少能区分 region-internal、经 local boundary 暂时对外可见以及硬外部可观察三类 live range。

#### Scenario: Local-boundary external live range is distinct from hard-external live range

- **WHEN** 一个 tile 写实例在 block 内只通过 `pto.treshape` 这类 local boundary 继续被后续链路消费
- **THEN** analysis MUST 将其标记为区别于硬外部可观察结果的 escape kind
- **AND** MUST NOT 把它与函数返回、terminator use、跨 block use 或等价硬外部 use 合并成同一类

#### Scenario: Region-internal temporary remains non-escaping

- **WHEN** 一个 tile 写实例只在当前 block-local 候选区域内被消费，且其生命周期在区域内结束
- **THEN** analysis MUST 将其标记为 non-escaping / internal
- **AND** 后续 region materialization 与 store-elision 阶段 MUST 可以直接消费该结论

### Requirement: PreFusionAnalysisPass MUST stay analysis-only while exposing store-elision inputs

这次扩展 MUST 只增强 analysis 元数据，不改变 planning / scheduling 的职责边界。

#### Scenario: Analysis exports more metadata without emitting transformation decisions

- **WHEN** `PreFusionAnalysisPass` 输出 write-instance 与 escape-class 信息
- **THEN** 它 MUST 继续保持 analysis-only
- **AND** MUST NOT 直接生成 `pto.fusion.group_id`、`pto.fusion.order`、`pto.yield` 或 store-elision 结果
- **AND** MUST 继续向后续 planning / region-generation / store-elision 提供可复用输入
