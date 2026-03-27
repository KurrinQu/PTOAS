## MODIFIED Requirements

### Requirement: A5 post-lowering fusion MUST run after `PTOToA5VM`

一旦 A5 backend 主线切到 `PTOToA5VM`，A5 fusion mainline MUST 固定按 `PTOFusionRegionGen -> PTOA5VMVersionSelection -> PTOToA5VM -> Canonicalizer -> PTOLowLevelLoopFusion -> CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion -> CSE` 顺序运行。  
其中，`PTOA5VMVersionSelection` MUST 位于 `PTOToA5VM` 之前，为 lowering 提供 per-op 决策；公共 `Canonicalizer` MUST 位于 `PTOToA5VM` 之后、`PTOLowLevelLoopFusion` 之前，用于删除残余常量条件控制流。  
同时，`PTOLowLevelLoopFusion` MUST 继续位于 `PTOToA5VM` 之后，并以 A5VM lowering 后的低层 loop 结构为输入契约。  
同时，在 `PTOToA5VM -> Canonicalizer -> PTOLowLevelLoopFusion -> CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion -> CSE` 阶段，地址模型 MUST 采用 memref-first 契约，不得为满足发射 ABI 提前退化为 pointer-only。

#### Scenario: Fusion mainline inserts version selection before lowering and cleanup before low-level fusion

- **WHEN** A5 backend 主线在 fusion 打开时执行 backend lowering 和 post-lowering 优化
- **THEN** `PTOA5VMVersionSelection` MUST 在 `PTOToA5VM` 之前运行
- **AND** 公共 `Canonicalizer` MUST 在 `PTOToA5VM` 之后、`PTOLowLevelLoopFusion` 之前运行
- **AND** `PTOLowLevelLoopFusion` MUST 继续以 `scf.for + a5vm.*` 低层结构作为正式输入
- **AND** region-preserving cleanup MUST 按 `CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion` 顺序运行
- **AND** MUST NOT 继续把旧 `pto.simd.vec_scope` / `vector.masked_*` bridge IR 当作该主线的正式输入契约
- **AND** MUST 在该阶段保持 memref-first 地址语义，直到进入发射边界
