## Context

### 范围

本 design 只覆盖 A5 Level3 OP-Lib 流水线中的 elementwise fusion 路径，目标 op 范围限定为：

- tile-tile binary:
  - `tmul/tdiv/tadd/tsub/tmax/tmin`
- tile-scalar:
  - `tmuls/tdivs/tadds/tsubs/tmaxs/tmins`

仅讨论以下 pass 链路：

`PTOCreateFusionGroups -> PTOOutlineFusionGroups -> PTOInstantiateAndLowerToLibCall -> PTOInlineLibCall -> PTOLowLevelLoopFusion -> PTOToEmitC`

### 当前状态

当前仓库已经具备以下基础能力：

- `PTOCreateFusionGroups` 能为一小部分连续 binary chain 打标，但规则仍是 binary-only。
- `PTOOutlineFusionGroups` 能把已打标的 group outline 成 `__pto_fused_group_*` helper，但 helper 接口仍假设固定三元 tile ABI。
- `PTOLowerToOpLibCalls` 与 `PTOInlineLibCall` 已经是 interface-driven lowering 路径，能够把 helper 内的 grouped op 变成 OP-Lib instance 并 inline。
- `PTOLowLevelLoopFusion` 已接在 inline 之后，但当前只匹配裸相邻双层 `scf.for`，无法跨 `pto.simd.vec_scope` 工作。
- `PTOToEmitC` 已支持 `pto.simd.vec_scope -> __VEC_SCOPE__` 的合法降级。

当前实现存在四个具体问题：

1. 分组能力过窄

- 只支持 binary-only chain，不能覆盖 tile-scalar 与 tile-tile 混合链。

2. outline helper 接口过窄

- 只围绕 `(src0, src1, dst)` 三元 tile 模型工作，无法稳定承载 scalar external input。

3. low-level fusion 入口不匹配 inline 后 IR

- inline 之后 fused helper 的规范形态是多个 entry-level `pto.simd.tile_to_memref` 和多个并列 `pto.simd.vec_scope` stage，而不是裸相邻 `scf.for`。

4. 缺少链内访存消除

- 当前每个 stage 都各自 `maskedload` 输入、计算、`maskedstore` 到中间 tile，再由后继 stage 再次读回。

### 实现约束

- 不改变现有 pipeline 顺序。
- 不新增用户可见 CLI 或公开 IR 入口。
- grouped lowering 继续复用现有 `OpLibOpInterface` 与 `OpLibMatchDescriptor`，不引入第二套 matcher 协议。
- `tdivs` 必须继续保留 `operandOrder` / `requiredVariantId` 约束，覆盖 `tile_scalar` 与 `scalar_tile`。
- 对无法证明 legality 的 case，`PTOLowLevelLoopFusion` 必须保守 no-op。

## Goals / Non-Goals

**Goals:**

- 把分组、outline、grouped lowering 的能力范围扩展到上述 12 个 elementwise op。
- 允许 tile-tile 与 tile-scalar 构成 mixed chain，只要 tile 数据依赖连续。
- 将 low-level fusion 的识别入口从裸 `scf.for` 升级到 inline 后的 `pto.simd.vec_scope` stage 形态。
- 在 fused helper 内消除仅供后继 stage 消费的中间 round-trip 访存。
- 用回归显式约束“单一 vec_scope + 单一 loop nest + 无链内中间访存”。

**Non-Goals:**

- 不扩展到 ternary / unary / reduction / broadcast family。
- 不引入跨多个 block、多个 region 或带 sync 的通用融合。
- 不改动 EmitC 的外部行为模型。

## Design

### 1. 分组与 outline 设计

#### 1.1 将 `PTOCreateFusionGroups` 升级为 descriptor-driven grouping

当前 `PTOCreateFusionGroups` 通过手写 `isFusibleBinaryOp/getBinarySrcs/getBinaryDst` 判断是否可分组。这一模型无法覆盖 tile-scalar，也无法统一处理 `tdivs` 的方向敏感语义。

本 change 将其升级为基于 `OpLibOpInterface` / `OpLibMatchDescriptor` 的分组逻辑：

- group eligibility 只接受上述 12 个 in-scope op。
- eligibility 通过 `OpLibOpInterface` 获取 descriptor，而不是 pass 内部继续堆积新的 `dyn_cast` 分发。
- chain continuation 规则：
  - 当前 op 的某个 tile 输入若消费前一 op 的 tile dst，则允许接入同组。
  - scalar 输入只作为 external input 记录，不影响 chain continuation。
  - `tdivs` 继续沿用 descriptor 中的 `operandOrder` 与 `requiredVariantId`，但这只影响后续 lowering，不改变“是否由 tile 依赖决定 chain”这一原则。

#### 1.2 mixed chain 的 group boundary

以下情况会打断 group：

- 非上述 12 个目标 op；
- region / block 边界；
- side-effect 或同步语义无法证明与当前链兼容的 op；
- 当前 op 不消费前一 op 的 tile dst；
- descriptor 不完整或无法稳定识别 tile/scalar 角色。

#### 1.3 扩展 `PTOOutlineFusionGroups` 的 helper 接口

当前 outline helper 假设 group interface 固定为三元 tile ABI。mixed chain 需要允许 scalar external input，同时允许 in-place tile-scalar op 出现在链中。

本 change 中 helper 接口调整为：

- `callArgs = 唯一 external operands + 按原顺序出现的 destination tiles`
- external operands 可以同时包含 tile 与 scalar
- 仍保留当前 helper 命名、`pto.fusion.group_id` attr 和 caller 中的单个 call boundary

为了降低维护成本，outline 阶段不再手写每个 op 的 clone 构造，而是基于统一的 operand remap 机制克隆原 op，自然保留：

- `tdivs` 的 `pto.tdivs.order`
- in-place 目的操作数
- 其他与 lowering/EmitC 相关的既有 attrs

### 2. Inline 后 canonical 形态与 low-level fusion

#### 2.1 维持现有 lowering / inline 顺序

`LowerToOpLibCall -> InlineLibCall` 顺序不变。grouped lowering 继续复用现有 interface-driven path，只扩展其对 mixed chain 的覆盖和一致性约束。

#### 2.2 将 `PTOLowLevelLoopFusion` 升级为 vec-scope-aware

当前 pass 只匹配裸相邻双层 `scf.for`。这与 OP-Lib inline 后的实际 helper 形态不一致，因此必须把识别入口升级为相邻 `pto.simd.vec_scope` stage。

v1 只支持 OP-Lib elementwise 模板的规范形态：

- helper entry 中允许存在若干 `pto.simd.tile_to_memref`
- 每个 stage 是单 block `pto.simd.vec_scope`
- `vec_scope` 内是单一 loop nest
- 各 stage 的 loop header 一致
- 各 stage 的 lane mask 构造一致
- stage 内只包含允许的 vector/arith/memref/scf 核心序列

对满足上述约束的相邻 stage，pass 需要重建为：

- 一个新的 `pto.simd.vec_scope`
- 一个单一 loop nest
- loop body 内按原顺序串接各 stage 的计算

不满足约束时保持 no-op。

### 3. 链内 load/store 消除

#### 3.1 store-to-load forwarding

在 fused loop body 内维护“中间 tile memref -> 当前 vector SSA 值”映射。

如果某个 stage 原本对中间 tile 执行：

- `vector.maskedstore` 写回
- 后继 stage 再从同一 tile `vector.maskedload`

则在 fused loop 中直接把后继 stage 的 load 改写为使用前序 stage 的 vector SSA 值。

#### 3.2 保留 store 的边界

只有两类 store 保留：

- group 最终输出所对应的 store
- 链外仍必须观察到的 in-place 结果

对于只在链内被后继 stage 消费的中间 tile，必须抑制 `vector.maskedstore`。

#### 3.3 `tile_to_memref` bridge 去重

entry-level `pto.simd.tile_to_memref` 允许保留，但对同一 helper 参数的重复 bridge 需要去重，避免在 fused helper 中为同一 tile 反复 materialize 等价 memref view。

## Testing Strategy

### 1. `test/tile_fusion/`

- 更新 `create_fusion_groups.mlir`
  - 覆盖 tile-scalar op 被标组
  - 覆盖 mixed chain 分组
- 更新 `materialize_fusion_groups.mlir`
  - 覆盖 mixed chain outline 后 helper 参数包含 scalar
- 重写 `low_level_loop_fusion.mlir`
  - 明确检查 fused helper 中只保留一个 `pto.simd.vec_scope`
  - 明确检查不存在链内中间 `vector.maskedstore` / 回读 `vector.maskedload`
  - 允许最终输出 store 保留
- 新增一个 negative test
  - 通过插入非目标 op 或制造不一致 loop shape，验证 low-level fusion 保守 no-op

### 2. `test/oplib/`

- 使用 `softmax_chain.pto` 作为 mixed chain 主验证
  - 验证 `tmuls/tmaxs/tmins` 能与后续 tile-tile binary 链共同进入融合路径
- 保留 `binary_max_min_chain.pto` 作为纯 tile-tile 对照 case
- 新增 1 个 EmitC/codegen smoke
  - 检查生成代码仍包含 `__VEC_SCOPE__`
  - 检查没有链内中间 tile 的多余 round-trip 访存模式

### 3. 验证命令

- `../llvm-project/build-shared/bin/llvm-lit -sv test/tile_fusion/...`
- `../llvm-project/build-shared/bin/llvm-lit -sv test/oplib/<相关用例>`
- `ptoas ... --enable-op-fusion -o %t.cpp` 的直出 C++ smoke

## Risks / Mitigations

- [Risk] 仅靠 descriptor 很难恢复 destination tile 与外部输入的边界
  - Mitigation：grouping 用 descriptor 判定 eligibility / 角色，outline helper 接口仍结合 op 原始 operands 与 DPS 目的操作数信息构建。

- [Risk] inline 后不同模板 stage 的 loop body 细节不完全一致，导致 low-level fusion 形态过窄
  - Mitigation：v1 明确只支持 OP-Lib elementwise 模板的规范形态；不满足时保持 no-op，并用 negative test 锁定保守行为。

- [Risk] 链内访存消除误删对链外可见的 in-place store
  - Mitigation：只删除“仅被后继 stage 消费”的中间 store；对最终输出和链外仍需可见的 destination 一律保留。

- [Risk] `tdivs` 方向敏感语义在 grouped lowering 中丢失
  - Mitigation：要求 grouped path 继续透传 `operandOrder` 和 `requiredVariantId`，并在 spec 与测试中单独点名。
