## 1. 重整 `ptoas` A5 backend pipeline

- [x] 1.1 重构 `tools/ptoas/ptoas.cpp` 中 A5 backend 分流，移除旧 `PTOInstantiateAndLowerToLibCall` / `PTOInlineLibCall` 作为 A5 主线 backend 的接线。
- [x] 1.2 固定 fusion 主线顺序，使 `createPTOFusionRegionGenPass()` 位于 `createLowerPTOToA5VMPass()` 之前，并让 `PTOFlattenFusionRegionPass` 成为 backend emission 之前的显式出口。
- [x] 1.3 保留 `EmitC` 输出路径，但移除其对旧 OP-Lib backend 主线和 `--op-lib-dir` 的正式依赖；同步整理相关 warning/flag 语义。

## 2. 改写 A5VM post-lowering low-level fusion

- [x] 2.1 更新 `PTOLowLevelLoopFusion` 的 pass 契约与文档声明，使其输入从旧 `pto.simd.vec_scope` / `vector.masked_*` 迁移到 `PTOToA5VM` 产出的 `scf.for + a5vm.*` 低层结构。
- [x] 2.2 实现 A5VM post-lowering `PTOLowLevelLoopFusion` 的匹配与保守融合逻辑，确保它位于 `PTOToA5VM` 之后时不再退化为 no-op。
- [x] 2.3 确认已封装 fusion group 在 `PTOToA5VM` / 新的 `PTOLowLevelLoopFusion` 中继续以 `pto.fusion_region` body 为原位变换边界，同时允许 `PTOToA5VM` 继续降低 region 外残余非融合 PTO op，并把 fusion group 的 region 生命周期保留到显式 flatten。

## 3. 更新回归与验证入口

- [x] 3.1 增加或改写 focused regression，锁定 A5 主线中 `PTOFusionRegionGen -> PTOToA5VM -> PTOLowLevelLoopFusion` 的顺序，以及旧 OP-Lib lowering passes 不再属于 A5 backend 主线。
- [x] 3.2 增加或改写 A5VM backend regression，验证 `pto.fusion_region` 在 A5VM lowering 与 post-lowering fusion 期间保持到 `PTOFlattenFusionRegionPass` 之前。
- [x] 3.3 增加或改写 A5 EmitC smoke regression，验证 `--pto-backend=emitc --pto-arch=a5` 不再要求旧 OP-Lib backend 主线即可完成输出。

## 4. 收尾验证与文档同步

- [x] 4.1 运行最小必要构建与 focused 测试，确认 `ptoas`、A5VM backend 和相关 tile fusion pass 顺序均通过回归。
- [x] 4.2 运行 `openspec validate replace-a5-oplib-lowering-with-a5vm-backend`，确保 change 工件结构完整、spec delta 合法。
