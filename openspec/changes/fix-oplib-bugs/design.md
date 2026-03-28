# Design: OpLib Bug 修复方案

## 详细设计 (Detailed Design)

### 1. 地址空间保留

- 在 `PTOInstantiateAndInlineOpLib` 中，克隆 `AllocTileOp` 时显式复制 `addr` 属性。

### 2. 步长推导增强

- 修改 `InferPTOLayout.cpp`，对于 `memref.subview` 转换而来的 PTO View 操作，计算步长。

### 3. 类型显式化

- 在 `PTOToEmitC` 中，生成必要的 C++ 类型转换代码。

## 交付物 (Deliverables)

- 增强后的 Layout 推导逻辑。
- 更稳健的属性克隆机制。
