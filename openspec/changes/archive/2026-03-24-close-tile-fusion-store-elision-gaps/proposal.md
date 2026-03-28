# Proposal: 补齐 tile fusion 冗余 store 消除链路

## 概述

当前 tile fusion 的 grouping、region 封装和 OP-Lib lowering 已经基本打通，但 `build/qwen.cpp` / `paged_attention_example_kernel_online_update.pto` 这类真实样例仍保留大量冗余 `vsts`。问题已经不在“能不能融合”，而在“从 `PreFusionAnalysis` 的 write-instance live range 到 `pto.yield` external frontier，再到低层 store 消除”的契约链路还没有闭合。

## 背景与动机

当前实现里存在四个直接相关的缺口：

1. `PreFusionAnalysis` 的 liveness 仍以底层 `Value` 为主键，DPS 复用同一个 destination tile 时，多个写实例会被折叠成一条 live range，无法稳定表达“同一块 tile storage 上的多次定义”。
2. `FusionPlanPass` / `OpSchedulingPass` 目前不消费 liveness 结果；`PTOFusionRegionGenPass` 虽然已经按“span 外是否还有 replaceable SSA use”构造 `pto.yield`，但当前 change 仍缺少“只把真正外部可见值编码进显式 frontier、并在后续 lowering/store-elision 中稳定保留这条边界”的收口。
3. grouped lowering 与 `PTOLowLevelLoopFusion` 只实现了很窄的同 base `vector.maskedstore -> vector.maskedload` forwarding。对 region 内最后一次写、但其实不再逃逸的临时 tile，没有消除机制。
4. `PTOFlattenFusionRegionPass` 之后只跑 `Canonicalizer + CSE`，没有针对 `vector.maskedstore` / `vector.store` 的 frontier-aware dead-store elimination，因此残余 store 会直接落到 EmitC/C++。

如果不先把这条链路的 contract 补齐，后续继续叠加 low-level 优化只会反复遇到同一类问题：analysis 无法准确描述 live range，region 无法稳定编码 yield frontier，lowering 侧也无法判断“哪些 store 真正可删”。

## 目标

- 新增正式的 `tile-fusion-analysis` capability，把 version-aware tile live range 和 escape/frontier 分类写成可验证契约。
- 新增正式的 `tile-fusion-store-elision` capability，把 lowering 后 `pto.fusion_region` 内的冗余 store 消除边界写成可验证契约。
- 修改 `tile-fusion-region-encapsulation`，要求 `pto.yield` / region result frontier 只承担“稳定顺序的显式外部可见值集合”这一职责，不再扩展 frontier-class metadata。
- 明确区分两类值：
  - region-internal dead temporary
  - 仍出现在 `pto.yield` frontier 中、需要在 v1 中保守保留的 yielded tile
- 在当前 change 中优先解决“region 内非逃逸 tail store 仍然残留”的问题；如果未来要对 `treshape` 特殊处理，应发生在 `FusionRegionGen` 之前，而不是通过扩展 region frontier contract 完成。

## 非目标

- 不在本 change 中重新定义 5.3 planning 或 5.4 scheduling 的成组合法性。
- 不在本 change 中把 `pto.treshape` 从 local non-through boundary 改成可直接穿透的 compute op。
- 不承诺在本 change 中完成跨 `pto.fusion_region`、跨 `pto.treshape` 的寄存器前传；这类 boundary-aware forwarding 如果需要，应在 `FusionRegionGen` 之前的阶段单独解决。
- 不修改 A3 路径，也不引入新的用户可见 CLI flag。

## 预期结果

- `PreFusionAnalysis` 能稳定区分“同一 tile storage 的不同写实例”，不再因为 DPS destination 复用而把多次定义压成一条 producer 记录。
- `PTOFusionRegionGenPass` 产出的 `pto.yield` frontier 只表达稳定顺序的 yielded value 列表，不再携带并行 frontier-class metadata。
- lowering 后新增或扩展的 store-elision 阶段能够删除 region 内不再逃逸的最后一次写入，而不是只删“后面正好又 load 回来”的 round-trip。
- 对仅通过 `pto.yield -> treshape` 暂时暴露到 region 外的 tile，v1 仍按普通 yielded frontier 保守保留；如果未来要进一步融合或前传，应在 `FusionRegionGen` 之前解决。
- `paged_attention_example_kernel_online_update.pto` 一类样例的回归可以精确区分：
  - 必须保留的 yielded/result store
  - 当前 v1 仍保守保留、经 `treshape` 暂时暴露的 yielded store
  - 本应消除但当前遗漏的 region-internal dead store

## 成功标准

- OpenSpec 中新增 `tile-fusion-analysis` 与 `tile-fusion-store-elision` 两个 capability。
- OpenSpec 中 `tile-fusion-region-encapsulation` 的 delta 明确要求 `pto.yield` frontier 只承担稳定顺序的显式 yielded value contract。
- 变更完成后，针对 DPS destination 复用的融合链，analysis dump 可以验证 distinct write instance 与正确的 escape 分类。
- 变更完成后，针对 `online_update` / `qwen` 这类样例，测试能够证明 region 内 non-escaping tail store 被删除，同时 yielded frontier 仍按约束保留。

## Capabilities

### New Capabilities

- `tile-fusion-analysis`: 规定 `PreFusionAnalysisPass` 的 version-aware tile live range、write instance 和 escape/frontier 分类语义。
- `tile-fusion-store-elision`: 规定 lowering 后 `pto.fusion_region` 内的 frontier-aware store 消除、保留与保守退化边界。

### Modified Capabilities

- `tile-fusion-region-encapsulation`: 将 `pto.yield` / `pto.fusion_region` result frontier 收敛为稳定顺序的显式 yielded value contract，供下游 store-elision 直接消费。

## Impact

- 影响分析与封装链路：
  - `include/PTO/Transforms/TileFusion/FusionAnalysis.h`
  - `lib/PTO/Transforms/TileFusion/FusionAnalysis.cpp`
  - `lib/PTO/Transforms/TileFusion/PTOFusionRegionGen.cpp`
- 影响 lowering / 消冗链路：
  - `lib/PTO/Transforms/TileFusion/PTOLowLevelLoopFusion.cpp`
  - 新增或扩展的 fusion-region store-elision pass
  - `tools/ptoas/ptoas.cpp` 中的融合流水线顺序
- 影响回归：
  - `test/tile_fusion/`
  - driver sample `test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto`
