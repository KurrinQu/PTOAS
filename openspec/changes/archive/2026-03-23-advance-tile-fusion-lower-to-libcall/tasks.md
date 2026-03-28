## 1. OpenSpec 契约落定

- [x] 1.1 编写 `proposal.md`，固定本 change 以 `pto.fusion_region` 为 5.5 之后正式输入边界，并明确不前移 `LowerToLibCall`。
- [x] 1.2 编写 `design.md`，收口 memref-world 过渡方案、region-transparent `PlanMemory` / `InsertSync`、region-based LowerToLibCall 与显式 flatten 的实现边界。
- [x] 1.3 编写 `specs/tile-fusion-region-lowering/spec.md` 与 `specs/oplib-lowering/spec.md`，定义 region 生命周期、grouped lowering 输入单元与 `trowexpandmul` scope。

## 2. `pto.fusion_region` 透明分析支持

- [x] 2.1 修改 `lib/PTO/Transforms/PTOPlanMemory.cpp` 及相关分析逻辑，使 `PlanMemory` 能递归进入 `pto.fusion_region` body，并把 `pto.yield` operands 与 region results 建立 alias / frontier 关系。
- [x] 2.2 修改 `lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp` 与相关 InsertSync 路径，使 `PTOInsertSync` 能递归处理 `pto.fusion_region`，并避免在 wrapper 本身和region内生成额外 sync boundary。
- [x] 2.3 增加 focused regression，验证 `fusion_region_interface.mlir` 一类用例不再因 `pto.fusion_region` wrapper 在 `PlanMemory` / `InsertSync` 失败。

## 3. Region-based LowerToLibCall

- [x] 3.1 修改 `lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp`，让 grouped path 直接以 `pto.fusion_region` 作为 lowering unit，而不是继续扫描已被 5.5 移除的 per-op `pto.fusion.*` metadata。同时把pass名也对齐文件名。
- [x] 3.2 保持 region-based grouped lowering 继续复用 `OpLibOpInterface` / `OpLibMatchDescriptor`，并对 partially-supported region 做 deterministic hard-fail。
- [x] 3.3 扩展 active grouped OpLib lowering scope 到 `trowexpandmul` family、`texp` family，并保证 row-broadcast 语义角色在 template / variant 选择前不丢失。

## 4. Region 生命周期与管线接线

- [x] 4.1 新增 `PTOFlattenFusionRegionPass`，在 `include/PTO/Transforms/Passes.h`、`include/PTO/Transforms/Passes.td` 与 `lib/PTO/Transforms/` 中完成声明、实现和注册。
- [x] 4.2 把 `PTOFlattenFusionRegionPass` 接到 `PTOLowLevelLoopFusion` 之后、Emit/手工降级之前，并确保 `PTOInlineLibCall` / `PTOLowLevelLoopFusion` 继续在 region body 内原位工作。
- [x] 4.3 更新 `tools/ptoas/ptoas.cpp` 的 A5 + fusion pipeline，固定新的 region 生命周期顺序并补充必要注释。

## 5. 回归与验证

- [ ] 5.1 更新 `test/tile_fusion/` 回归，覆盖 region-transparent `PlanMemory` / `InsertSync`、region-based LowerToLibCall 和 flatten 后无 residual `pto.fusion_region`。
- [ ] 5.2 新增或更新 `test/oplib/` / `test/tile_fusion/` 用例，覆盖 `trowexpandmul -> trowexpandmul -> tadd` 正例，以及 partially-supported region hard-fail 负例。
- [ ] 5.3 运行 focused `lit` 验证与至少一个 `ptoas ... --enable-op-fusion --pto-arch=a5` smoke，记录 `pto.fusion_region` 能经过 LowerToLibCall 并在 Emit 前被消解。
