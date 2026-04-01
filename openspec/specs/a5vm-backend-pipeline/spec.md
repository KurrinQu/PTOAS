# a5vm-backend-pipeline Specification

## Purpose
定义 A5 fusion mainline 中从 `PTOFusionRegionGen` 到 post-lowering cleanup、loop fusion 与 flatten 的固定 backend pipeline 契约，以及该阶段必须保持的 memref-first 地址语义。
## Requirements
### Requirement: A5 post-lowering fusion MUST run after `PTOToA5VM`

一旦 A5 backend 主线切到 `PTOToA5VM`，A5 fusion mainline MUST 固定按 `PTOFusionRegionGen -> PTOA5VMVersionSelection -> PTOToA5VM -> PTOA5VMIfCanonicalize -> PTOLowLevelLoopFusion -> CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOPostFusionLoopUnroll -> PTOFlattenFusionRegion -> CSE` 顺序运行。  
其中，`PTOA5VMVersionSelection` MUST 位于 `PTOToA5VM` 之前，为 lowering 提供 per-op 决策；`PTOA5VMIfCanonicalize` MUST 位于 `PTOToA5VM` 之后、`PTOLowLevelLoopFusion` 之前，并且只处理 residual `scf.if` cleanup，不得把 `scf.for` loop-shape canonicalization 混入该阶段。  
同时，`PTOLowLevelLoopFusion` MUST 位于 `PTOToA5VM` 之后，并以 A5VM lowering 后的低层 loop 结构为输入契约。  
同时，在 fusion mainline 打开时，region-preserving cleanup 顺序 MUST 固定为 `CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOPostFusionLoopUnroll -> PTOFlattenFusionRegion -> CSE`，不得跳过新增的 post-fusion unroll 阶段，也不得把它挪到 flatten 之后。  
同时，在 `PTOToA5VM -> PTOA5VMIfCanonicalize -> PTOLowLevelLoopFusion -> CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOPostFusionLoopUnroll -> PTOFlattenFusionRegion -> CSE` 阶段，地址模型 MUST 采用 memref-first 契约，不得为满足发射 ABI 提前退化为 pointer-only。

#### Scenario: Fusion mainline inserts version selection before lowering and cleanup before low-level fusion

- **WHEN** A5 backend 主线在 fusion 打开时执行 backend lowering 和 post-lowering 优化
- **THEN** `PTOA5VMVersionSelection` MUST 在 `PTOToA5VM` 之前运行
- **AND** `PTOA5VMIfCanonicalize` MUST 在 `PTOToA5VM` 之后、`PTOLowLevelLoopFusion` 之前运行
- **AND** `PTOA5VMIfCanonicalize` MUST 只 canonicalize residual `scf.if` 结构，不得改写周围 `scf.for` loop header
- **AND** `PTOLowLevelLoopFusion` MUST 继续以 `scf.for + a5vm.*` 低层结构作为正式输入
- **AND** region-preserving cleanup MUST 按 `CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOPostFusionLoopUnroll -> PTOFlattenFusionRegion -> CSE` 顺序运行
- **AND** MUST NOT 继续把旧 `pto.simd.vec_scope` / `vector.masked_*` bridge IR 当作该主线的正式输入契约
- **AND** MUST 在该阶段保持 memref-first 地址语义，直到进入发射边界

#### Scenario: Low-level loop fusion consumes A5VM low-level IR under memref-first addressing

- **WHEN** A5 backend 主线在 fusion 打开时执行 post-lowering 优化
- **THEN** `PTOLowLevelLoopFusion` MUST 在 `PTOToA5VM` 之后运行
- **AND** MUST 以 `scf.for + a5vm.*` 低层结构作为正式输入
- **AND** region-preserving cleanup MUST 按 `CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOPostFusionLoopUnroll -> PTOFlattenFusionRegion -> CSE` 顺序运行
- **AND** MUST NOT 继续把旧 `pto.simd.vec_scope` / `vector.masked_*` bridge IR 当作该主线的正式输入契约
- **AND** MUST 在该阶段保持 memref-first 地址语义，直到进入发射边界

#### Scenario: Post-fusion loop unroll remains a fixed pre-flatten stage even when it chooses no-op

- **WHEN** `PTOPostFusionLoopUnroll` 针对某个 fusion-local carrier loop 判定当前不值得展开
- **THEN** 实现 MAY 对该 loop 保持 no-op
- **AND** 但主线顺序 MUST 仍然保留 `PTOFusionLoadStoreElision -> PTOPostFusionLoopUnroll -> PTOFlattenFusionRegion`
- **AND** MUST NOT 因为该 loop 没有被展开就把该阶段挪到 flatten 之后或提前到 pre-lowering 阶段

## ADDED Requirements

### Requirement: A5VM emission stage MUST bridge memref-first IR to pointer ABI

在 A5 backend 采用 memref-first 主线的前提下，A5VM 发射阶段 MUST 负责完成必要的 pointer ABI 对接，不得反向要求前段 pass 放弃 memref 地址语义。

#### Scenario: Emission boundary performs ABI bridging without changing semantics

- **WHEN** A5VM IR 以 memref 地址形态进入 text/LLVM 发射路径
- **THEN** 发射阶段 MUST 将地址参数桥接为目标 intrinsic 所需 pointer ABI
- **AND** MUST 保持与等价 pointer 输入相同的调用语义
