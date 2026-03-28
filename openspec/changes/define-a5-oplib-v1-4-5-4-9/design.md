## 范围

本 change 定义 A5 OpLib V1 的设计边界，范围只覆盖 `PTO_IR_manual.md` 第 4.5~4.9 节：

- 4.5 Vector Arithmetic Operations
- 4.6 Reduction Operations
- 4.7 Broadcast Operations
- 4.8 Compare & Select Operations
- 4.9 Bitwise Operations

V1 的 in-scope op 集固定如下：

- 4.5 Binary tile-tile:
  - `tadd`, `tsub`, `tmul`, `tdiv`, `tmax`, `tmin`, `trem`
- 4.5 Partial / shape-policy arithmetic:
  - `tpartadd`, `tpartmax`, `tpartmin`, `tprelu`
- 4.5 Tile-scalar:
  - `tadds`, `tsubs`, `tmuls`, `tdivs`, `tmaxs`, `tmins`, `trems`
- 4.5 Ternary:
  - `taddc`, `tsubc`, `taddsc`, `tsubsc`
- 4.5 Unary:
  - `tabs`, `tneg`, `texp`, `tlog`, `tsqrt`, `trsqrt`, `trecip`, `trelu`, `tlrelu`
- 4.6 Reduction:
  - `trowsum`, `trowmax`, `trowmin`, `tcolsum`, `tcolmax`, `tcolmin`
- 4.7 Broadcast:
  - `trowexpand`, `tcolexpand`, `trowexpandmul`, `trowexpanddiv`, `trowexpandsub`, `texpands`
- 4.8 Compare & Select:
  - `tcmp`, `tcmps`, `tsel`, `tsels`
- 4.9 Bitwise:
  - `tand`, `tor`, `txor`, `tshl`, `tshr`, `tnot`

以下内容明确不属于 V1：

- `tload`、`tstore`、`tmov`、`ttrans`、`tsync`
- `tmatmul` / `tgemv` 等 matrix compute
- 4.5~4.9 之外的 PTO IR manual 内容
- 新的 Level-1 / Level-2 作者接口
- 在 `pto-isa` 仓内承载 MLIR compiler 基础设施

## 当前状态

当前 PTOAS 已经具备以下基础：

- PTO dialect、`pto.simd.*`、`PTOLowerToOpLibCalls`、`PTOValidateSimdIR`、`PTOToEmitC`
- `oplib/level3/*.mlir` concrete 模板
- `oplib/level3/generate_level3_templates.py`
- 一批基于 A5 vector 路径的 lit 回归

当前模板维护方式仍然是：

- `skeletons/catalog.json`
- `*.instance.tmpl.mlir`
- Python 生成器做字符串替换
- 根目录输出 importer-active concrete `.mlir`

这个状态已经能覆盖一部分 family，但还有三个结构性问题：

1. 作者接口偏文本替换

- `catalog`、模板占位符和 concrete 输出之间缺少统一的结构化 schema。
- 新增 family 时，生成器需要堆叠更多专用拼接逻辑。

2. 覆盖范围和语义来源没有统一 manifest

- 目前缺少“IR manual 首批范围 + pto-isa A5 实际支持矩阵 + PTOAS 已实现状态”的统一真值文件。
- 很难回答某个 op 是“V1 应实现但尚未完成”，还是“`pto-isa` A5 本身尚无稳定语义，应该 deferred”。

3. 编译器消费侧与作者侧耦合方式偏脆弱

- importer 仍然只消费 concrete `.mlir`，这点本身没问题，但当前 concrete 生成过程难以对 family 级约束做完整静态表达。
- 后续要补 ternary、partial binary、reduce/broadcast 等 family 时，继续沿用字符串模板会把维护复杂度进一步放大。

## 设计拆分

### 1. 工程落点与职责边界

V1 的实现主战场固定在 `PTOAS`：

- `PTOAS` 负责：
  - Family DSL
  - 模板生成器
  - OP-Lib lowering / validation / EmitC
  - A5 manifest 对齐和回归
- `pto-isa` 负责：
  - A5 语义真源
  - 头文件实现和现有后端约束
  - 作为 PTOAS 的自动对齐输入

这样做的理由：

- 当前 PTOAS 已经有完整 MLIR/Pass/EmitC 基础设施，继续在这里承载 V1 风险最低。
- `pto-isa` 当前主要是 API/header/backend 仓，不适合在 V1 阶段再引入第二套编译器基础设施。

### 2. 作者接口从文本 skeleton 升级为声明式 Family DSL

V1 作者接口固定为两层输入：

1. Family spec

- 定义 `kind`
- 定义参数角色和 ABI 形状
- 定义支持的 op 列表
- 定义支持的 dtype 轴
- 定义 compare condition / variant / scalar position / `isBinary` 等匹配轴
- 定义 family 级默认 metadata、cost、priority、A5 legality
- 定义与 `pto-isa` A5 manifest 的映射键

2. Mixed-Body MLIR snippet

- 只表达核心 vector 计算体
- 不负责外围循环、tail mask、`tile_to_memref`、统一命名、metadata 和 concrete `func.func` 拼装

V1 不再把以下内容暴露给库作者手工维护：

- concrete `func.func` 的符号命名
- 全量 `pto.oplib.*` 属性样板
- 外围 `scf.for` / `vector.create_mask` / `vector.maskedload` / `vector.maskedstore` 样板
- 仅因 `dtype`、`opname`、condition 或 variant 变化而复制的完整函数体

### 3. Generator 负责的固定框架

生成器职责固定，不留给实现阶段自由发挥：

- 读取 Family spec 和 snippet
- 校验 spec/schema
- 补齐 concrete `func.func`
- 生成：
  - 参数列表
  - `pto.oplib.kind`
  - `pto.oplib.entry_role`
  - `pto.oplib.op`
  - `pto.oplib.variant_id`
  - `pto.oplib.match.*`
  - 统一的 `tile_to_memref`
  - `pto.simd.vec_scope`
  - tail mask
  - 固定 64-lane vector load/store 骨架
  - concrete `.mlir` 输出
- 提供 `--write` 和 `--check`

生成器必须继续输出 checked-in concrete `.mlir`，原因如下：

- 兼容现有 `--op-lib-dir` importer
- 便于 review、debug、lit 和 CI diff
- 不要求 `PTOLowerToOpLibCalls` 直接读取 DSL 源

### 4. Snippet 合同固定化

为防止每个 family 又变成一套独立微语言，V1 固定 snippet 合同：

- Binary / tile-scalar / unary / ternary family：
  - snippet 读取约定的 vector SSA 名称
  - snippet 产出 `%result`
- Compare family：
  - snippet 产出 `%cmp`
- Select family：
  - snippet 产出 `%result`
- Reduction / broadcast family：
  - snippet 只写核心规约/扩展计算
  - loop、accumulator 初始化、tail mask、final store 由生成器固定提供

这样可以让作者只关心核心语义，不反复手写外围 SIMD scaffolding。

### 5. Family 划分

V1 family 划分固定如下：

- `float_binary_elementwise`
- `int_binary_elementwise`
- `partial_binary_elementwise`
- `float_tile_scalar`
- `int_tile_scalar`
- `ternary_elementwise`
- `float_unary`
- `int_unary_bitwise`
- `reduce_row`
- `reduce_col`
- `broadcast_row`
- `broadcast_col`
- `broadcast_row_binary`
- `scalar_expand`
- `cmp_tile_tile`
- `cmp_tile_scalar`
- `select_mask`
- `select_scalar_mode`

这份划分要服务于 4.5~4.9 全量 op，而不是只覆盖当前 catalog 已有 family。

### 6. `pto-isa` A5 自动对齐 manifest

V1 新增 A5 自动对齐真值层：

- PTOAS 内新增一个 manifest snapshot，记录 4.5~4.9 op 在 A5 上的状态
- snapshot 来源于从 `pto-isa` 自动提取的结果
- manifest 至少包含：
  - op 名
  - 所属 manual section
  - family
  - A5 状态：`implemented` / `deferred`
  - 支持 dtype
  - 关键 layout / tmp / scalar / mask / variant 约束
  - 对应 `pto-isa` 语义来源路径

使用规则固定如下：

- `implemented`：
  - V1 需要生成 concrete 模板
  - 需要有 lowering 测试
- `deferred`：
  - 不生成 lowering 候选
  - 需要有显式原因
  - 不允许静默缺位

### 7. Lowering 契约

`PTOLowerToOpLibCalls` 在 V1 中需要满足：

- 对 4.5~4.9 in-scope op 强制执行 OpLib lowering 逻辑
- 使用 concrete 模板，不直接消费 DSL 源
- 保持现有 matcher key 语义不变：
  - `kind`
  - `op`
  - `dtype`
  - `variant_id`
  - `cmpMode`
  - `scalarPos`
  - `requiredVariantId`
  - `isBinary`
- 对 `implemented` 但无候选实例的情况硬失败
- 对 `deferred` op 给出确定性诊断
- Pass 结束后不得在普通用户函数中残留 in-scope PTO op

### 8. 不默认新增新 public `pto.simd` op

V1 默认复用已有集合：

- `pto.simd.tile_to_memref`
- `pto.simd.vec_scope`
- `pto.simd.predicate`
- `pto.simd.load/store/load_pu/store_pu`
- `vector.*`
- `arith.*`
- `memref.*`
- `scf.*`
- `math.*` 中当前已允许的子集

只有当某个 4.5~4.9 op 无法在这套集合内表达时，才允许引入新的最小扩展；一旦引入，必须同步补 verifier、lowering 和 EmitC 规则。

## 实现约束

- `--op-lib-dir` 输入模型不变，仍指向 concrete 模板目录。
- concrete `.mlir` 继续作为 checked-in 文件存在。
- 作者入口仍是 Level-3，不新增 Level-1/Level-2 公共作者 DSL。
- V1 不在 `pto-isa` 仓中增加 MLIR 构建入口。
- 64-lane SIMD 继续是 V1 唯一前向约束。
- 动态 shape 只覆盖动态 valid shape，不扩展到动态 physical tile shape。
- 每个 in-scope op 的 dtype/layout 合法性以 `pto-isa` A5 manifest 为准，不允许 PTOAS 私自扩张。
- compare/select/bitwise 这类已有独立 capability 的 op，V1 可以复用现有语义，但不能与已有 spec 矛盾。

## 测试策略

测试按四层组织：

1. 生成器测试

- Family spec 解析
- snippet 合同校验
- concrete 输出稳定性
- `--check` 漂移检测

2. lowering / verifier 负测

- 缺失必要 `pto.oplib.*` 属性
- family / signature / variant 不匹配
- `implemented` op 无候选模板
- `deferred` op 误被当成已实现使用

3. family 级 lit 回归

- 4.5 arithmetic、ternary、unary、tile-scalar
- 4.6 reduction
- 4.7 broadcast
- 4.8 compare/select
- 4.9 bitwise

4. 对齐测试

- IR manual 4.5~4.9 标题集合与 manifest 对齐
- manifest 与 `pto-isa` A5 自动提取结果对齐
- `implemented` op 必须存在 concrete 模板和 lowering 回归

## 风险与缓解措施

- 风险：`pto-isa` A5 语义提取无法完全自动化
  - 缓解：manifest 允许 `deferred`，但必须显式列出原因和来源路径。

- 风险：partial / ternary / broadcast family 的 snippet 合同过于僵硬
  - 缓解：先定义 family 固定合同，再按 family 扩展最小字段，不允许每个 op 自己发明新约定。

- 风险：保留 concrete `.mlir` 会增加生成产物数量
  - 缓解：这是为 importer 兼容和 review 可读性做出的明确 trade-off，V1 不优化为运行时生成。

- 风险：旧生成器和新 DSL 在迁移期并存，维护负担暂时上升
  - 缓解：V1 文档要求新 in-scope family 只走新 DSL；旧生成器只为迁移期兜底，不继续扩面。

- 风险：compare-bitwise 等已有能力与新 V1 规则重叠
  - 缓解：V1 只补 authoring / lowering 契约，不重写已有 op 语义 spec。
