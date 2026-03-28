## MODIFIED Requirements

### Requirement: A5 post-lowering fusion MUST run after `PTOToA5VM`

一旦 A5 backend 主线切到 `PTOToA5VM`，`PTOLowLevelLoopFusion` MUST 位于 `PTOToA5VM` 之后，并以 A5VM lowering 后的低层 loop 结构为输入契约。  
同时，在 fusion mainline 打开时，region-preserving cleanup 顺序 MUST 固定为 `CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion`，不得跳过新增的 predicate-elision 阶段，也不得把它挪到 flatten 之后。  
同时，在 `PTOToA5VM -> PTOLowLevelLoopFusion -> CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion` 阶段，地址模型 MUST 采用 memref-first 契约，不得为满足发射 ABI 提前退化为 pointer-only。

#### Scenario: Low-level loop fusion consumes A5VM low-level IR under memref-first addressing

- **WHEN** A5 backend 主线在 fusion 打开时执行 post-lowering 优化
- **THEN** `PTOLowLevelLoopFusion` MUST 在 `PTOToA5VM` 之后运行
- **AND** MUST 以 `scf.for + a5vm.*` 低层结构作为正式输入
- **AND** region-preserving cleanup MUST 按 `CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion` 顺序运行
- **AND** MUST NOT 继续把旧 `pto.simd.vec_scope` / `vector.masked_*` bridge IR 当作该主线的正式输入契约
- **AND** MUST 在该阶段保持 memref-first 地址语义，直到进入发射边界
