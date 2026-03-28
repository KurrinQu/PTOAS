# a5vm-backend-pipeline Specification

## MODIFIED Requirements

### Requirement: A5 post-lowering fusion MUST run after `PTOToA5VM`

一旦 A5 backend 主线切到 `PTOToA5VM`，`PTOLowLevelLoopFusion` MUST 位于 `PTOToA5VM` 之后，并以 A5VM lowering 后的低层 loop 结构为输入契约。  
同时，在 `PTOToA5VM -> PTOLowLevelLoopFusion -> PTOFlattenFusionRegion` 阶段，地址模型 MUST 采用 memref-first 契约，不得为满足发射 ABI 提前退化为 pointer-only。

#### Scenario: Low-level loop fusion consumes A5VM low-level IR under memref-first addressing

- **WHEN** A5 backend 主线在 fusion 打开时执行 post-lowering 优化
- **THEN** `PTOLowLevelLoopFusion` MUST 在 `PTOToA5VM` 之后运行
- **AND** MUST 以 `scf.for + a5vm.*` 低层结构作为正式输入
- **AND** MUST NOT 继续把旧 `pto.simd.vec_scope` / `vector.masked_*` bridge IR 当作该主线的正式输入契约
- **AND** MUST 在该阶段保持 memref-first 地址语义，直到进入发射边界

## ADDED Requirements

### Requirement: A5VM emission stage MUST bridge memref-first IR to pointer ABI

在 A5 backend 采用 memref-first 主线的前提下，A5VM 发射阶段 MUST 负责完成必要的 pointer ABI 对接，不得反向要求前段 pass 放弃 memref 地址语义。

#### Scenario: Emission boundary performs ABI bridging without changing semantics

- **WHEN** A5VM IR 以 memref 地址形态进入 text/LLVM 发射路径
- **THEN** 发射阶段 MUST 将地址参数桥接为目标 intrinsic 所需 pointer ABI
- **AND** MUST 保持与等价 pointer 输入相同的调用语义
