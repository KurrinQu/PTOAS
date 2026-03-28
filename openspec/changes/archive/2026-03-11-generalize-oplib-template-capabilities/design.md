# Design: 多 Family Level-3 OP-Lib 基础设施

## 设计目标

本设计要解决的不是某一个 op 家族，而是 OP-Lib 框架本身的 binary-only 假设。

目标是提供一层足够窄、但能支撑 4.4-4.8 的公共底座：

1. family-specific `kind`
2. variable-arity 模板签名
3. per-argument 匹配元数据
4. family-specific attr 匹配
5. 更宽的模板体 IR 白名单
6. 更宽的 A5 OP-Lib vector EmitC 支持矩阵

## `kind` 与签名矩阵

legacy `l3_binary_elementwise_template` 保持兼容。

新增 family 在注册阶段按 `kind` 绑定固定签名类别：

1. `(tile, tile, dst)`
2. `(tile, scalar, dst)`
3. `(tile, tile, tile, dst)`
4. `(tile, scalar, tile, dst)`
5. `(tile, dst)`
6. `(scalar, dst)`
7. `(src, tmp, dst)`
8. `(mask, src0, src1, dst)`
9. `(src0, src1, selectMode, dst)`

签名校验按 `kind` 驱动，不再由全局 `validateTemplateSignature()` 固定判断。

## 匹配元数据

### 兼容层

legacy binary family 继续支持：

1. `pto.oplib.match.rows`
2. `pto.oplib.match.cols`
3. `pto.oplib.match.blayout`
4. `pto.oplib.match.slayout`
5. `pto.oplib.match.fractal`

### 新元数据

新 family 一律改用按参数编号的形式：

1. `pto.oplib.match.argN.rows`
2. `pto.oplib.match.argN.cols`
3. `pto.oplib.match.argN.blayout`
4. `pto.oplib.match.argN.slayout`
5. `pto.oplib.match.argN.fractal`

额外 attr 匹配：

1. `pto.oplib.match.scalar_pos`
2. `pto.oplib.match.cmp_mode`
3. `pto.oplib.match.is_binary`

这些属性只在对应 family 上解释，不做全局强制。

## Pass 泛化

### 注册与校验

`TemplateRegistry` 需要从固定字段模型扩成 family-aware 结构：

1. 记录 `kind`
2. 记录参数类别序列
3. 记录 `argN.*` 匹配元数据
4. 记录可选 attr 匹配元数据

### 匹配

matcher 从“取 binary interface 后匹配”改为：

1. 先识别 IR op 所属 family
2. 生成统一的 family match request
3. 逐项比较 operand class、`argN.*`、dtype、layout、attr 条件

### 实例化与 call rewrite

instance key 不再拼固定 3 个参数类型，而是拼：

1. `variant_id`
2. 所有 concrete argument type
3. family-specific attr choices

call rewrite 也不再使用 binary helper，而是按 family 统一重写。

### Inline

`PTOInlineLibCallPass` 不再保留 binary-only 形态的隐式假设，实例函数参数个数和 operand order 必须完全由实例符号签名决定。

## 模板体 IR 白名单

在现有 `arith/vector/memref/scf` 基础上，新增最小 `math.*` 集合：

1. `math.exp`
2. `math.log`
3. `math.sqrt`
4. `math.rsqrt`

不放开其他 `math` op。

## A5 OP-Lib vector EmitC 扩展

当前 A5 OP-Lib 只支持 float vector load/store 与 float binary arith。

本 change 需要补齐后续 family 所需的最小能力：

1. float unary vector lowering
2. `math.exp/log/sqrt/rsqrt` 对应的 vector lowering
3. vector compare / select lowering
4. vector reduction lowering
5. integer vector load/store legality检查
6. integer vector bitwise / shift lowering

## 模板源码目录统一

模板源码目录统一到 `oplib/level3/`：

1. lit 直接用 `--op-lib-dir=<repo>/oplib/level3`
2. 删除 `test/tile_fusion/oplib/` 下的重复模板源
3. 负测资源仍保留在 `test/tile_fusion/resources/`

## 测试

本 change 自身只补基础设施测试：

1. 非法 family 签名
2. 缺失 `argN.*`
3. 非法 `scalar_pos/cmp_mode/is_binary`
4. 模板体出现未放行的 `math` / compare / reduction / integer vector op
5. A5 vector dtype/lanes 非法
6. lit 切换到 `oplib/level3/` 后原 binary regression 继续通过
