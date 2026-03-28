# Design: 补全缺失算子的 OpLib 支持

## 详细设计 (Detailed Design)

### 1. OpLib 模板定义 (oplib/level3/)

需要在 `oplib/level3/` 目录下提供对应的 MLIR 模板文件。

- **`trem` / `trems`**: 在 `int_binary_elementwise_templates.mlir` 或 `float_binary_elementwise_templates.mlir` 中添加模板，或者根据数据类型放入相应的模板组。
- **`tprelu` / `tlrelu`**: 在激活函数相关的模板文件中添加实现。对于 `tlrelu` (Leaky ReLU)，模板需要能够接收 `slope` 作为参数。

### 2. 算子映射 (PTOLowerToOpLibCalls.cpp)

在 `PTOLowerToOpLibCalls.cpp` 中新增降低模式（Pattern）：

- **`TRemOp`**: 映射到 `trem` 模板。
- **`TRemSOp`**: 映射到 `trems` 模板，处理标量操作数。
- **`TPReluOp`**: 映射到 `tprelu` 模板，处理特征图和权重 Tile。
- **`TLReluOp`**: 映射到 `tlrelu` 模板，将 `slope` 属性作为常量操作数或属性传递给模板。

### 3. 实例化与内联

利用现有的 `PTOInstantiateAndInlineOpLib` 机制，根据输入的 Shape 和 Type 自动特化模板并进行代码内联。

## 交付物 (Deliverables)

- `oplib/level3/` 下新增或修改的模板文件。
- `PTOLowerToOpLibCalls.cpp` 中的映射逻辑更新。
- 端到端测试用例 `test/oplib/missing_ops_oplib.mlir`。
