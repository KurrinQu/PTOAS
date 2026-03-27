# Proposal: 引入 A5VM per-op 版本选择 pass

## 概述

当前 A5 fusion mainline 中，`PTOToA5VM` 同时承担 PTO op 版本选择和 A5VM codegen 两类职责。`post_update/no_post_update` 与 `1D/2D` 的选择逻辑散落在 lowering 内部，容易把策略分支直接编码成 `scf.if`，导致 pass 责任边界不清，也让后续对版本策略的演进持续侵入 `PTOToA5VM`。

本 change 计划把版本选择前移为 `PTOFusionRegionGen` 之后的独立 pass，通过 PTO op 属性显式记录每个 A5VM candidate op 的 lowering 决策；`PTOToA5VM` 只消费该决策并直接生成选定版本，lowering 后再直接调用公共 `Canonicalizer` 清理残余常量条件控制流。

## 背景与动机

当前实现暴露出三个直接问题：

1. `PTOToA5VM` 破坏面过大

- 版本选择与 codegen 模板耦合在同一处。每次调整 `post_update/no_post_update` 或 `1D/2D` 选择策略，都容易把行为判断、模板分叉和 fallback 一起改动，放大对 `PTOToA5VM` 主逻辑的影响面。

2. 选择结果不显式

- 现阶段版本选择主要隐含在 lowering 逻辑内部，pass 边界之间没有正式的“已选定版本”契约。下游看不到每个 PTO op 被判定为哪一类版本，也无法在 `PTOToA5VM` 之前对选择结果做单独验证。

3. 条件分支混入正式 lowering 结果

- 当版本选择依赖 lowering 内部条件时，IR 中会出现用于分支选择的 `scf.if` / `else`。这让后续 post-lowering pass 既要处理真正需要的低层控制流，也要处理仅仅因为版本决策而残留的常量条件分支，增加 fusion 和 cleanup 的负担。

当前阶段更合理的边界是：先独立完成 per-op 版本选择，再让 `PTOToA5VM` 直接按既定版本生成 codegen，并把常量条件清理视为后置优化，而不是主要决策机制。

## 目标

- 新增 `PTOFusionRegionGen` 之后的 A5VM 版本选择 pass，为每个 A5VM candidate PTO op 显式记录 lowering 决策。
- 版本选择范围覆盖全部 A5VM candidate PTO op，包括 `pto.fusion_region` 内部 op 和 residual PTO op。
- 为 PTO dialect 定义正式的 per-op 版本决策属性，至少覆盖 `post_update/no_post_update` 与 `1D/2D` 两类选择维度。
- 要求 `PTOToA5VM` 直接读取 per-op 决策并生成选定版本，不再把版本选择条件作为主要 IR 结构输出。
- 在 `PTOToA5VM` 之后新增 cleanup 阶段，直接调用公共 `Canonicalizer` 删除残余常量条件 `if/else`，保持后续 `PTOLowLevelLoopFusion` 的输入更干净。
- 补齐 OpenSpec 契约和回归测试，明确 pipeline 顺序、属性边界、fail-fast 行为和兼容性范围。

## 非目标

- 不在本 change 中重做 `post_update/no_post_update` 或 `1D/2D` 的判定算法，只重构决策位置和决策载体。
- 不把 region 级默认配置或跨 op 联合选择引入为本轮正式接口。
- 不修改 A3 路径或 A5 之外 backend 的 lowering 契约。
- 不在 fusion mainline 中继续保留 `--a5vm-lowering-strategy` 作为正式决策输入。

## 预期结果

- `PTOFusionRegionGen` 之后的 IR 中，每个 A5VM candidate PTO op 都带有明确的 per-op A5VM lowering 选择属性。
- `PTOToA5VM` 的职责收敛为“消费选择结果并生成 A5VM IR”，不再承担主要版本判定逻辑。
- fusion mainline 中与版本选择有关的 `scf.if` / `else` 不再作为正式 lowering 结果保留；若仍有残余常量条件控制流，由 `Canonicalizer` cleanup 阶段在 `PTOLowLevelLoopFusion` 前清理。
- `pto.fusion_region` 继续作为结构化 lowering 边界，版本选择属性不会改变 region frontier 和 flatten 责任。

## 成功标准

- OpenSpec 新增 `a5vm-version-selection` capability，明确 per-op 属性、选择 pass 和 `PTOToA5VM` 消费契约。
- OpenSpec 修改 `a5vm-backend-pipeline`，把 A5 fusion mainline 的固定顺序更新为 `PTOFusionRegionGen -> 版本选择 -> PTOToA5VM -> Canonicalizer cleanup -> PTOLowLevelLoopFusion -> ...`。
- OpenSpec 修改 `tile-fusion-region-lowering`，明确 `pto.fusion_region` 内外 PTO op 都可携带版本选择属性，且不改变 region 边界。
- 回归测试能够证明 `PTOToA5VM` 对已标注 op 直接生成选定版本，并在缺失或非法属性时 fail-fast。

## What Changes

- 新增 `PTOA5VMVersionSelectionPass`，在 `PTOFusionRegionGen` 之后为全部 A5VM candidate PTO op 标注 per-op lowering 决策。
- 在 PTO dialect 中新增 A5VM lowering 选择属性，采用单一复合 typed attribute 同时承载 `update_mode` 和 `loop_shape`。
- 修改 `PTOToA5VM` 契约，使其在 fusion mainline 中直接消费 per-op 决策并按选定版本 codegen。
- 在 `tools/ptoas/ptoas.cpp` 中于 `PTOToA5VM` 之后直接调用公共 `Canonicalizer`，移除残余常量条件控制流。
- 调整 A5 fusion mainline pipeline 顺序，并补充相应 lit / pipeline 回归。
- 在 fusion mainline 中废弃 `--a5vm-lowering-strategy` 的决策语义，仅保留非 fusion 路径兼容性。

## Capabilities

### New Capabilities
- `a5vm-version-selection`: 定义 A5VM per-op 版本选择属性、版本选择 pass、`PTOToA5VM` 消费与校验契约。

### Modified Capabilities
- `a5vm-backend-pipeline`: 更新 A5 fusion mainline 的固定 pass 顺序，并约束 `PTOToA5VM` 前后新增的版本选择与 `Canonicalizer` cleanup 阶段。
- `tile-fusion-region-lowering`: 补充 `pto.fusion_region` 内外 PTO op 携带版本选择属性时的结构化 lowering 边界契约。

## Impact

- 受影响代码主要位于 `include/PTO/IR/`、`include/PTO/Transforms/`、`lib/PTO/Transforms/`、`tools/ptoas/ptoas.cpp` 和相关 `test/` 回归。
- 受影响接口包括 PTO dialect attribute 定义、A5VM backend pipeline 接线、`PTOToA5VM` 的决策输入形式，以及 fusion mainline 下 `--a5vm-lowering-strategy` 的 CLI 语义。
- 对现有下游 pass 的正向影响是：`PTOLowLevelLoopFusion`、`PTOFusionPredicateElision`、`PTOFusionLoadStoreElision` 将接收到更稳定、少无关分支的 A5VM lowering 结果。
