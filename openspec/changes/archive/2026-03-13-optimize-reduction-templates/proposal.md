# Proposal: 优化 Reduction 模板以减少代码冗余

## 目标 (Goals)

通过引入“种子系统” (Seed System)，消除 OpLib 中 Reduction 类算子（Sum, Max, Min 等）的大量重复代码。

## 现状 (Current State)

目前每个 Reduction 算子都有独立的 `.mlir` 模板，但其内部的 Tiling 逻辑、循环结构和同步机制几乎完全一致。

## 预期行为 (Expected Behavior)

1. 创建通用的 Reduction 框架模板。
2. 定义核心运算的“种子函数”接口。
3. 通过模板实例化机制，将不同的种子注入通用框架。

## 影响范围 (Scope)

- `oplib/level3/reduction_templates.mlir`
- `lib/PTO/Transforms/PTOInstantiateAndInlineOpLib.cpp`
