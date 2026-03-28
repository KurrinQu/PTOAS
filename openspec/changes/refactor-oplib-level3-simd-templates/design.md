## Context

当前 `oplib/level3` 模板存在三类结构性问题：

1. SIMD 约束不一致

- 同一 family 内同时存在 32-lane 和 64-lane实现，`f32/i32` 路径大量使用 `vector<32x...>`，`f16/i16/i8` 路径则混有 `vector<64x...>`。
- `lib/PTO/Transforms/PTOToEmitC.cpp` 已将 A5 OP-Lib data vector 建模为固定 64 lanes，但仍保留对历史总位宽模板的兼容路径，造成模板端和后端端的长期分叉。

2. 模板复用方式低效

- 当前模板按 `op`、`dtype`、condition 拆成大量平行文件或平行函数，主体循环骨架、load/store 路径和 mask 逻辑重复。
- compare family 以 `LT/LE/GT/GE/EQ/NE` 分别编码，bitwise/tile-scalar/unary family 也存在只改核心算子或元素类型的重复体。

3. 模板体风格混杂

- 若干模板仍通过 `memref.load/store` 做逐元素实现，例如 `float_binary_elementwise_f16_scalar_templates.mlir`、`float_broadcast_row_templates.mlir`、`int_unary_i16_scalar_templates.mlir`。
- `PTOLowerToOpLibCalls.cpp` 对模板体仍保留标量 `memref.load/store` 兼容口，导致“SIMD 编程范式”无法形成硬约束。

本 change 的职责不是直接实现重构，而是确定后续实现必须遵循的模板架构、Lowering 契约和验证边界。

## Goals / Non-Goals

**Goals:**

- 将 `oplib/level3` 模板体系统一到 64-lane SIMD 模型。
- 定义“单一 skeleton source + 多个 concrete dtype/condition 实例”的模板组织方式，替代当前手工平铺模板。
- 明确非标量相关 family 不再接受 `memref.load/store` 逐元素实现。
- 保持现有 lowering 匹配键语义，包括 `dtype`、`cmpMode`、`scalarPos`、`requiredVariantId` 和 `isBinary`。
- 保持 `rows/cols/v_row/v_col` 在模板匹配与类型语义中继续使用 `i64` 口径。

**Non-Goals:**

- 不设计单个 `func.func` 级别的真泛型模板机制。
- 不改变 PTO IR 类型系统的 shape/validShape 基础存储类型。
- 不在本 change 内重新定义 OP 语义或扩张当前 OP-Lib lowering 目标集合。
- 不要求一次性解决所有 A5 backend 能力缺口；模板覆盖仍受“OP 本身支持的 dtype”与后端支持矩阵限制。

## Decisions

### Decision 1: 采用“单一 skeleton source + concrete 实例展开”而不采用手工多份模板

采用单一 skeleton source 维护每一类计算模式的公共结构，包括：

- tile/mask/scalar 参数位置
- 64-lane vector load/store 形式
- mask 计算与 tail 处理
- core slot 或 condition slot
- 统一的 shape / layout / fractal 匹配元数据骨架

由 skeleton source 展开生成 concrete 实例，实例维度至少包括：

- dtype 轴：`i8/i16/i32`，以及 OP 支持时的 `f16/f32`
- condition 轴：compare family 的 `LT/LE/GT/GE/EQ/NE`
- variant 轴：如 `tdivs` 的 `tile_scalar` / `scalar_tile`

选择该方案的原因：

- 满足“同一计算模式使用同一套模板”的目标。
- 保留当前 OP-Lib 匹配与实例化对 concrete `func.func` 的依赖，不强行引入新的泛型 IR 机制。
- 比“为每个 dtype/condition 手写一份模板”更易维护，比“运行时改写单个泛型模板”实现风险更低。

备选方案：

- 手工维护多份 concrete 模板：实现最简单，但无法满足复用目标，拒绝。
- 扩展 `PTOLowerToOpLibCalls.cpp` 使其在实例化时执行跨 dtype retype：复用更强，但会显著扩大 Pass 改造范围，首轮不采用。

### Decision 2: 非标量相关 family 一律采用纯 SIMD 数据路径

对于非标量相关 family，模板体仅允许：

- `vector.*`
- `arith.*`
- `scf.*`
- 受限 `memref.*`，但不再允许 `memref.load/store` 作为数据通路
- `pto.simd.tile_to_memref`
- 必要的 `pto.simd.*`

标量相关 family 的定义收敛为 ABI 上显式带 scalar operand 的模板类别，例如：

- `l3_float_tile_scalar_template`
- `l3_int_tile_scalar_elementwise_template`
- `l3_float_ternary_tile_scalar_template`
- `l3_scalar_expand_template`
- `l3_select_scalar_template`

即便是这些 family，也只允许：

- 标量参数作为 builtin scalar 输入
- 使用 `vector.splat` 或等价 SIMD 方式并入 64-lane 计算

不再允许通过 `memref.load/store` 对 tile 进行逐元素标量回退。对 `trowexpand`、`trowexpandmul` 等当前内部抽取 row-scalar 的 family，后续实现必须改写为仍处于 SIMD allowlist 内的方案。

选择该方案的原因：

- 与“必须使用 SIMD 编程范式”的用户要求一致。
- 避免模板导入规则继续为历史标量模板开例外。
- 简化模板校验和 EmitC 路径假设。

备选方案：

- 保留 row/broadcast family 的标量 `memref.load/store` 例外：过渡成本低，但会永久保留第二套模板范式，拒绝。

### Decision 3: 64-lane 作为 Level-3 模板的唯一前向约束

后续实现中，Level-3 模板以 64 lanes 作为唯一前向约束：

- 数据 vector 必须是 `vector<64xT>`
- mask vector 必须与 64-lane 数据 vector 对应
- 不再为 32-lane 模板新增能力

Lowering / EmitC 可以在迁移期保留兼容旧模板的分支，但新建或重构后的 Level-3 模板必须全部满足 64-lane 规则。

选择该方案的原因：

- 与现有 A5 方向保持一致。
- 为 skeleton source 统一提供单一设计点，避免继续出现 `f32/i32` 走 32-lane、`f16/i16/i8` 走 64-lane 的分叉。

备选方案：

- 按元素宽度允许 32/64/128 lanes 混用：会重新引入模板爆炸，不采用。

### Decision 4: compare 条件与 dtype 作为模板展开维度，而不是文件组织维度

compare family 的 `LT/LE/GT/GE/EQ/NE` 由同一 skeleton source 表达，condition 通过展开维度映射到 concrete 实例。类似地，同一计算模式支持的 dtype 也通过展开维度映射到 concrete 实例。

Lowering 侧继续使用既有匹配键：

- compare 仍通过 `cmpMode`
- dtype 仍通过 `pto.oplib.match.dtype`
- variant 仍通过 `variant_id`

即，模板源统一，但导入后的实例仍保持 concrete `func.func` 和 concrete 元数据，不改动现有 matcher 的键空间。

选择该方案的原因：

- 不破坏当前 `MatchRequest` 和 `TemplateRegistry::selectVariantFor` 的核心建模。
- 允许实现阶段按最小风险重构模板，而不需要同步重写所有 lowering 匹配逻辑。

### Decision 5: `rows/cols/v_row/v_col` 继续使用 `i64`

本 change 明确保留 `rows/cols/v_row/v_col` 的 `i64` 口径：

- `TileBufType` 的 `shape/validShape` 继续使用现有 `int64_t` 容器
- OP-Lib 模板匹配元数据中的 `rows/cols/fractal` 继续沿用 `i64`
- 相关测试与模板声明不引入新的 `i32` 维度约束

选择该方案的原因：

- 与现有 PTO IR 和 OP-Lib 匹配实现保持一致。
- 避免把 `oplib/level3` 模板重构扩大为 PTO 类型系统修改。

## Risks / Trade-offs

- [Risk] skeleton source 与生成出的 concrete 模板发生漂移 → Mitigation：后续实现阶段引入单一生成入口，并将生成结果纳入测试或一致性检查。
- [Risk] 64-lane 统一后，现有 32-lane `f32/i32` 模板在 EmitC 或 backend 路径上暴露新问题 → Mitigation：实现阶段按 family 分批迁移，并补 `test/oplib` emitc/generic-shape 回归。
- [Risk] row/broadcast family 去掉 `memref.load/store` 后实现复杂度上升 → Mitigation：先在设计中限定“禁止逐元素 scalar fallback”，实现时允许使用 `vector.extract`、`vector.shuffle` 或等价 SIMD 手段完成标量语义。
- [Risk] compare condition 与 dtype 同时收敛到统一模板源后，symbol 命名和实例键更复杂 → Mitigation：保持导入后的 concrete `variant_id`、`dtype`、`cmpMode` 语义不变，仅改变模板源组织。
- [Risk] 迁移期间仓库中会同时存在旧模板和新模板 → Mitigation：任务拆分中要求先补 importer 校验与测试，再分 family 切换模板源，避免部分 family 长期处于混合状态。

## Migration Plan

1. 在 OpenSpec 层落定 `oplib-templates` 与 `oplib-lowering` 契约。
2. 实现阶段先收紧模板导入校验，明确 64-lane 和 `memref.load/store` 规则。
3. 引入 skeleton source 到 concrete 实例的生成机制，并优先迁移重复度最高的 family：
   - compare family
   - int binary / int tile-scalar family
   - float binary / float tile-scalar family
4. 逐步替换 `oplib/level3` 中现有手写重复模板，并同步更新 `test/oplib`。
5. 在全部目标 family 迁移完成后，再移除对旧模板组织方式的兼容依赖。

回退策略：

- 若某个 family 在 64-lane 迁移中暴露 backend 不兼容，可临时保留旧 concrete 模板，但不回退已落地的 skeleton source 方案与导入契约。

## Open Questions

- skeleton source 的落盘位置是直接放在 `oplib/level3/` 下，还是引入独立目录/生成脚本目录；该问题会影响实现期的文件组织，但不影响本 change 的需求边界。
- 迁移期是否保留旧 concrete 模板文件并行存在，还是分 family 一次性切换；该问题属于实施顺序选择，不影响最终契约。
