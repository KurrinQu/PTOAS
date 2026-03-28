## 1. Analysis 数据模型

- [x] 1.1 扩展 `include/PTO/Transforms/TileFusion/FusionAnalysis.h` 与 `lib/PTO/Transforms/TileFusion/FusionAnalysis.cpp`，为 `PreFusionAnalysis` 新增 write-instance 与 escape-class 元数据。
- [x] 1.2 在 `buildPreFusionAnalysis()` 中实现 DPS destination 复用场景的 distinct write-instance 跟踪，避免后写覆盖前写的 producer/liveness 记录。
- [x] 1.3 更新 `lib/PTO/Transforms/TileFusion/PTOPrintPreFusionAnalysis.cpp` 的稳定文本输出，能观察 write-instance 与 escape classification。

## 2. Region Frontier 契约

- [x] 2.1 修改 `lib/PTO/Transforms/TileFusion/PTOFusionRegionGen.cpp`，让 `pto.yield` frontier 只承担稳定顺序的显式 yielded value 契约，不再携带 frontier class。
- [x] 2.2 保证 internal-dead tile 不会进入 `pto.yield` / region results，也不会生成伪 frontier entry。
- [x] 2.3 检查并补齐 `PTOViewToMemref`、`PTOInstantiateAndLowerToLibCall`、`PTOInlineLibCall`、`PTOLowLevelLoopFusion` 对显式 yielded frontier 的保留路径。

## 3. Yielded-Frontier Load/Store Elision

- [x] 3.1 新增或扩展专门的 fusion-region load/store-elision pass，并在 `tools/ptoas/ptoas.cpp` 中接到 `PTOLowLevelLoopFusion` 之后、`PTOFlattenFusionRegionPass` 之前；当前 `PTOLowLevelLoopFusion` 中已有的 load elimination / round-trip forwarding 逻辑一并迁移到该 pass。
- [x] 3.2 在该 pass 中统一处理同 base/index/mask 的 `vector.maskedstore -> vector.maskedload` / 等价 load round-trip 消除，不再由 `PTOLowLevelLoopFusion` 负责这部分 low-level forwarding。
- [x] 3.3 在该 pass 中删除 non-escaping tile write-instance 对应的最终 `vector.maskedstore` / 等价 store，不再要求必须存在后继同 base reload；对所有仍出现在 `pto.yield` / `region results` 中的 yielded frontier 保持保守保留，不做未经证明的跨 boundary forwarding，`treshape` 特殊处理放在 `FusionRegionGen` 之前考虑。

## 4. 回归与验证

- [x] 4.1 新增 `test/tile_fusion/pre_fusion_analysis_*.mlir` 覆盖 DPS destination 复用、write-instance 与 escape-class dump。
- [x] 4.2 新增或更新 `test/tile_fusion/fusion_region_*.mlir` / store-elision 用例，覆盖 yielded frontier 稳定顺序、internal-dead omission 和 non-escaping tail store 删除。
- [x] 4.3 对 `test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto` 增加 focused regression，验证 internal-dead store 被消除、仍出现在 yielded frontier 中的 store 按 v1 约束保留。
- [x] 4.4 运行最小相关验证：`python3 -m lit -sv test/tile_fusion`，并记录 sample smoke / 仍保守保留的剩余 store 场景。
