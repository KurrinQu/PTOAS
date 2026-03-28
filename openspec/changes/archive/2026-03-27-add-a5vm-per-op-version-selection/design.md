## Context

### 范围

本 design 只覆盖 A5 fusion mainline 中从 `PTOFusionRegionGen` 到 `PTOFlattenFusionRegion` 之前的 backend lowering 主线，不扩展到：

- A3 路径
- 非 fusion 的其他 backend
- `PTOToA5VM` 之后低层优化 pass 的通用重构

本轮只处理两个版本选择维度：

- `post_update` / `no_post_update`
- `1D` / `2D`

### 当前状态

当前 `tools/ptoas/ptoas.cpp` 中的 A5 fusion mainline 为：

`PTOFusionRegionGen -> PTOToA5VM -> PTOLowLevelLoopFusion -> CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion -> CSE`

当前实现有三个结构性问题：

1. 选择逻辑与 codegen 耦合

- `PTOToA5VM` 通过全局 `--a5vm-lowering-strategy` 和 lowering 内部条件共同决定具体模板，pass 既做版本判定，又做 codegen。

2. 版本决策不可见

- `PTOFusionRegionGen` 之后的 PTO IR 没有“每个 op 被选成什么版本”的正式契约，无法在 `PTOToA5VM` 前单独检查或测试版本决策。

3. 选择性分支污染 post-lowering IR

- 当 lowering 内部为兼容多种版本而构造 `scf.if` / `else` 时，后续 low-level loop fusion 和 cleanup 阶段会同时面对“真正业务控制流”和“仅用于版本选择的控制流”。

### 约束

- `pto.fusion_region` 必须继续作为结构化 lowering 边界，直到 `PTOFlattenFusionRegion`。
- residual PTO op 允许继续在 `pto.fusion_region` 外部直接降低，本轮不能为了统一版本选择而补包 synthetic region。
- `PTOLowLevelLoopFusion` 仍然必须在 `PTOToA5VM` 之后运行，且继续消费 lowering 后的 A5VM 低层 loop 结构。
- 非 fusion 路径当前仍依赖 `--a5vm-lowering-strategy`，本轮不能无条件删掉该兼容入口。

## Goals / Non-Goals

**Goals:**

- 将 A5VM 版本选择从 `PTOToA5VM` 中拆出，形成独立、可测试、可诊断的 pass。
- 以 PTO dialect typed attribute 显式承载 per-op 版本选择结果，覆盖 `pto.fusion_region` 内外全部 A5VM candidate PTO op。
- 让 `PTOToA5VM` 在 fusion mainline 中直接按 per-op 决策生成目标版本，而不是继续通过 `if/else` 做主选择。
- 在 `PTOToA5VM` 之后增加 cleanup 阶段，直接调用公共 `Canonicalizer` 清除残余常量条件控制流，稳定后续 low-level loop fusion 输入。
- 保持现有主线的 region-preserving 契约和 memref-first 地址语义不变。

**Non-Goals:**

- 不重做 `1D/2D` 或 `post_update/no_post_update` 的算法判定规则。
- 不引入 region 级默认策略、函数级策略或跨 op 联合调度策略。
- 不在本 change 中统一收敛所有 `PTOToA5VMLowering.cpp` 内部辅助函数的实现风格。
- 不修改 A5VM 发射阶段、EmitC 路径或 pointer ABI bridging 契约。

## Decisions

### 1. 用独立 pass 产生 per-op 决策，而不是继续在 `PTOToA5VM` 内部隐式判定

决策：

- 新增 `PTOA5VMVersionSelectionPass`
- 运行位置固定在 `PTOFusionRegionGen` 之后、`PTOToA5VM` 之前
- pass 粒度固定为 `func::FuncOp`
- pass 必须覆盖函数体内所有 A5VM candidate PTO op，不区分其是否位于 `pto.fusion_region` 内

原因：

- 版本选择本质上是 lowering 前的策略决策，独立出来后可单独测试、单独诊断、单独演进。
- `func::FuncOp` 粒度与 `PTOFusionRegionGen`、`PTOFlattenFusionRegion` 一致，便于同时处理 region 内外 residual op。

备选方案：

- 继续在 `PTOToA5VM` 内部做选择：实现最省事，但会继续扩大 `PTOToA5VM` 责任面，已被需求否决。
- 把选择信息挂在 `pto.fusion_region` 上：不能覆盖 residual PTO op，也会引入 region 默认值和 per-op override 的二次复杂度。

### 2. 版本选择结果使用单一复合 typed attribute，而不是多个零散属性

决策：

- 在 PTO dialect 中新增 `pto.a5vm_lowering_choice`
- 属性采用单一复合 typed attribute
- 属性中固定包含两个字段：
  - `update_mode = post_update | no_post_update`
  - `loop_shape = one_d | two_d`

原因：

- 两个维度必须作为同一个 lowering 决策被校验和消费；分散成两个属性会把完整性检查、默认值处理和错误诊断拆散。
- PTO dialect 已有 typed attribute 和 enum attr 先例，复用现有 attribute 体系比临时 string attr 更稳。

备选方案：

- 两个独立属性：实现门槛低，但不利于表达“必须同时存在且组合合法”的契约。
- `DictionaryAttr` 或 string attr：短期快，但类型约束弱，错误只能在运行期零散兜底。

### 3. `PTOToA5VM` 在 fusion mainline 中只消费显式决策，并对缺失决策 fail-fast

决策：

- fusion mainline 中，`PTOToA5VM` 对每个 A5VM candidate PTO op 必须读取 `pto.a5vm_lowering_choice`
- 若属性缺失、字段缺失或组合非法，pass 必须立即报错并 `signalPassFailure`
- lowering helper 按属性直接 dispatch 到对应 body，不再把版本选择条件作为主要 IR 结构输出

原因：

- 直接消费显式决策可以把错误最早暴露在 lowering 边界，而不是在后续 optimizer 或 emitter 中以间接形式暴露。
- fail-fast 能保护新契约，不让未标注或错误标注的 op 静默回退到旧行为。

备选方案：

- 缺失属性时自动回退到旧的 `--a5vm-lowering-strategy`：兼容性更强，但会削弱新 pass 的正式性，并掩盖版本选择遗漏。
- 仍允许 lowering 内部再做一轮策略重判：会形成“双重真相源”，需求已经明确不接受。

### 4. 在 backend pipeline 中直接调用 `Canonicalizer` 删除残余常量条件控制流

决策：

- 不新增专用 cleanup 阶段实现
- 固定在 `PTOToA5VM` 之后、`PTOLowLevelLoopFusion` 之前直接调用公共 `Canonicalizer`
- cleanup 目标仍然是残余常量条件 `scf.if` / `else` 与由其产生的死分支

原因：

- 版本选择迁出后，`PTOToA5VM` 理论上不再主动产生版本选择型 `if/else`，但仍可能保留来自其他模板或保守 lowering 的常量分支。
- 在 low-level loop fusion 之前先清理死分支，可以减少模式匹配噪声，且不打破现有 region-preserving cleanup 顺序。
- 直接复用公共 canonicalization 机制比为单一 backend cleanup 需求额外维护新 pass 更符合当前 driver 接线和用户约束。

备选方案：

- 仅依赖其他已有优化而不在 pipeline 中显式插入 `Canonicalizer`：不保证这一步稳定出现在 backend 契约中。
- 把 cleanup 放到 `PTOLowLevelLoopFusion` 之后：会让 loop fusion 继续面对额外分支，违背本 change 的目标。

### 5. fusion mainline 中废弃 `--a5vm-lowering-strategy` 的决策语义，非 fusion 路径继续兼容

决策：

- 当 fusion mainline 打开时，`--a5vm-lowering-strategy` 不再参与正式 lowering 决策
- driver 保留该 CLI 以兼容现有接口，但需要在 fusion mainline 下给出 ignored warning
- 非 fusion 路径暂不改变，继续沿用现有全局策略

原因：

- 新方案的单一真相源应是 per-op 属性；如果 CLI 仍能覆盖或回退，就会重新引入第二套决策入口。
- 保留参数本身可以避免无关路径被本 change 一并打断。

备选方案：

- 全局删除 CLI：破坏非 fusion 现有用法，超出本 change 范围。
- 允许 CLI 覆盖 per-op 决策：便于调试，但会直接破坏 spec 契约和测试稳定性。

### 6. 规格拆分采用“新 capability + 两个已有 capability 增量修改”

决策：

- 新增 `a5vm-version-selection`
- 修改 `a5vm-backend-pipeline`
- 修改 `tile-fusion-region-lowering`

原因：

- 版本选择本身是新的 backend 能力边界，适合独立建 capability。
- pipeline 顺序和 `pto.fusion_region` 结构边界已经在现有 spec 中定义，只需要在原 spec 上补充新约束。

## Testing Strategy

- 为版本选择 pass 增加 lit 测试，验证 `pto.fusion_region` 内外目标 op 都会被正确标注 `pto.a5vm_lowering_choice`
- 为 `PTOToA5VM` 增加 direct-emission 回归，检查已标注 op 只生成被选中的版本，不再输出用于版本选择的 `scf.if`
- 为 `PTOToA5VM` 增加 fail-fast 回归，覆盖缺失属性和非法组合属性
- 为 `Canonicalizer` cleanup 阶段增加回归，验证常量条件分支被清理且语义保持
- 为完整 fusion mainline 增加集成回归，验证新增 pass 顺序与 `pto.fusion_region` 边界契约不冲突
- 为非 fusion 路径保留兼容性回归，验证 `--a5vm-lowering-strategy` 旧行为不受本 change 影响

## Risks / Trade-offs

- [Risk] 现有 `PTOToA5VMLowering.cpp` 的版本选择逻辑分散，前移为独立 pass 时容易遗漏边缘 case
  - Mitigation：v1 复用现有合法性判断，先迁移现有决策，不同时重写策略算法；通过缺失属性/非法组合负例测试兜底。

- [Risk] per-op typed attribute 会新增 PTO dialect IR 面积，后续若扩展更多维度可能触发接口演进
  - Mitigation：v1 只纳入已确认的两个维度，并把属性命名和字段语义写入 spec，后续扩展走显式 schema 演进。

- [Risk] fail-fast 会让部分过去“能走但不稳定”的 case 提前失败
  - Mitigation：这是有意行为，用来暴露未被选择 pass 覆盖的 case；proposal 和 spec 明确把它定义为契约升级而非 regression。

- [Risk] `Canonicalizer` cleanup 若匹配面超出预期，可能错误折叠非版本选择相关的真实控制流
  - Mitigation：v1 只匹配常量条件 `scf.if`，不做一般化 CFG 简化；测试中加入保守不匹配 case。

## Migration Plan

1. 先补 OpenSpec，锁定 capability、pass 顺序和 CLI 兼容边界。
2. 增加 PTO dialect attribute 与 transform pass 声明，再接线 `tools/ptoas/ptoas.cpp`。
3. 先实现 `PTOA5VMVersionSelectionPass` 和属性读取，再改 `PTOToA5VM` 为 direct-emission。
4. 在 fusion mainline 中把公共 `Canonicalizer` 插入到 `PTOToA5VM` 后。
5. 补齐 lit 和 pipeline 集成测试，最后根据测试结果收敛 warning/diagnostic 文案。

回退策略：

- 若实现中发现部分 op 的选择逻辑无法在本轮稳定迁移，可临时只对已覆盖 op 启用选择 pass，并让未覆盖 op 在 fusion mainline 中显式报错，而不是静默回退旧路径。

## Open Questions

- 当前无阻塞性开放问题。
- 后续若需要增加第三维版本信息，例如更细的 template family 选择，应以扩展 `pto.a5vm_lowering_choice` 的 typed schema 为准，而不是重新引入独立 string attr。
