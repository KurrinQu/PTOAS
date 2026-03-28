# Proposal: 为 A5VM `vld*/vst*` 引入 memref-first 双地址形态

## 概述

当前 A5VM `vld*/vst*` 家族在 ODS 层主要按裸指针建模，编译器在中后段难以稳定保留 tile 相关 shape/stride/memory space 语义，导致 load/store 消除等优化需要额外恢复地址语义。  
本 change 引入 `memref + ptr` 双地址形态：编译主线优先使用 memref，保持结构化信息；同时保留裸指针形态，供 lower 边界与手写 A5VM 细粒度编程使用。

## 背景与动机

当前实现存在两个直接问题：

1. 对编译器优化不友好
- 裸指针地址形态会丢失 tile 的结构化信息，优化阶段需要通过额外分析补回语义，放大实现复杂度与不确定性。

2. 工程目标存在双诉求
- 编译主线需要 memref 语义以支持优化；同时 A5VM 作为低层 IR 仍需要保留裸指针入口，兼容已有下游和手工编程路径。

如果继续保持 pointer-only 的类型契约，后续在 fusion/消除/调度阶段会持续出现“优化需要结构化地址语义，但接口只暴露裸指针”的重复问题。

## 目标

- 为 `vld*/vst*`（含无状态与 predicate 变体）建立同名双形态地址契约：`memref` 与 `!llvm.ptr` 均可作为地址操作数。
- 将 A5 backend 主线约束为 memref-first：在 `PTOToA5VM -> PTOLowLevelLoopFusion -> PTOFlattenFusionRegion` 阶段默认保留 memref 地址语义。
- 明确 emit/LLVM 边界职责：在发射阶段完成必要 pointer ABI 映射，不要求前段提前放弃 memref 语义。
- 保留裸指针形态并持续兼容，不引入 *_mr 之类新 op。
- 明确 stateful store 变体边界：`pstu/vstu/vstus/vstur` 的 `base/base_out` 继续维持指针语义。

## 非目标

- 不扩展 `copy_* / gather / scatter` 等非 `vld*/vst*` 家族。
- 不新增用户可见 CLI 选项或 backend 切换行为。
- 不在本 change 中定义 stateful `base_out` 的 memref post-update view 语义。
- 不改变 A5 之外架构路径。

## 能力变更

### New Capabilities

- `a5vm-vld-vst-addressing`：定义 `vld*/vst*` 双地址形态、memref-first 主线、stateful 指针边界与发射等价性契约。

### Modified Capabilities

- `a5vm-backend-pipeline`：补充 A5 backend 主线的 memref-first 地址模型约束，以及发射阶段 pointer ABI 对接边界。

## 预期结果

- 编译器在 A5 主线优化阶段可直接消费 memref 地址语义，减少针对 pointer-only IR 的补偿逻辑。
- 手写 A5VM 或低层路径仍可继续使用裸指针，不破坏现有可用性。
- 发射结果在语义上保持一致：memref/ptr 两种输入形态不会导致不同的内建调用行为。

## 成功标准

- OpenSpec 中新增 `a5vm-vld-vst-addressing` capability，并覆盖双形态、主线默认、stateful 边界与发射契约。
- OpenSpec 中对 `a5vm-backend-pipeline` 的增量明确 memref-first 阶段约束与 pointer ABI 边界职责。
- 变更后测试可同时验证：
  - `vld*/vst*` 的 memref 形态通过；
  - 同一批 op 的 ptr 形态保持兼容；
  - stateful `base/base_out` 的 memref 输入按契约拒绝（负例）。
