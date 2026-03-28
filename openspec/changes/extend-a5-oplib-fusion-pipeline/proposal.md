# Proposal: 扩展 A5 OpLib 融合流水线契约

## 概述

当前 PTOAS 已经接通 A5 Level3 OP-Lib 的基础融合流水线：

`MarkFusionGroup -> OutlineFusionGroup -> LowerToOpLibCall -> LibCall Inline -> Low-Level loop fusion -> EmitC`

但现阶段该链路仍缺少可验收的变换契约。实际 fused helper 在 `PTOInlineLibCall` 之后仍然保留多个并列的 `pto.simd.vec_scope`，`PTOLowLevelLoopFusion` 对这些 helper 基本仍是 no-op，链内中间 tile 也会保留多余的 `vector.maskedstore` / `vector.maskedload` 往返。现有测试主要证明“helper 被生成”和“arithmetic IR 出现”，还不足以证明真正发生了 vec-scope-aware 的 low-level fusion。

## 背景与动机

当前 review 已确认三个直接影响 OP 融合可用性的问题：

1. `vec_scope` 阻断 low-level loop fusion

- `PTOLowLevelLoopFusion` 目前只匹配裸相邻的双层 `scf.for`，对 inline 后由多个 `pto.simd.vec_scope` 包裹的 elementwise stage 不生效。

2. 融合后链内 load/store 未消除

- 即使多个 OP 已经进入同一个 fused helper，inline 后每个 stage 仍各自执行 `tile_to_memref + vec_scope + maskedload/maskedstore`，导致中间 tile 反复写回再读回。

3. 回归无法证明“真的融合了”

- 现有 `test/tile_fusion/` 和 `test/oplib/` 主要检查 helper 存在、call 被替换或 arithmetic op 出现，没有把“单一 vec_scope”、“中间访存消除”和 mixed chain 覆盖作为硬约束。

如果继续在没有明确 contract 的情况下直接实现融合优化，后续很容易出现：

- mixed chain 是否允许分组没有统一口径；
- grouped lowering 与 single-op lowering 的能力边界不一致；
- low-level fusion 看似接入、实际持续 no-op；
- 测试仍然无法阻止 regression。

## 目标

- 为 A5 Level3 OP-Lib 融合链路补全 OpenSpec 契约，覆盖 `MarkFusionGroup -> OutlineFusionGroup -> LowerToOpLibCall -> LibCall Inline -> Low-Level loop fusion -> EmitC` 全流程。
- 将 fusion group 的目标范围扩展到 6 个 tile-tile binary op 与 6 个 tile-scalar 版本：
  - `tmul/tdiv/tadd/tsub/tmax/tmin`
  - `tmuls/tdivs/tadds/tsubs/tmaxs/tmins`
- 明确 mixed chain 合法性：tile-tile 与 tile-scalar 只要数据依赖连续即可进入同一 fusion group，scalar 作为外部输入不打断分组。
- 要求 `PTOLowLevelLoopFusion` 变成 `vec_scope` aware，能够把 inline 后的多个 elementwise stage 收敛为单个 `pto.simd.vec_scope` + 单个 loop nest。
- 要求 fused helper 链内消除仅供后继 stage 使用的中间 `vector.maskedstore` / `vector.maskedload` 往返。
- 补齐能够证明“真的融合了”的 IR lit 回归与 EmitC/codegen smoke。

## 非目标

- 不扩展到 ternary、unary、reduction、broadcast 等 family。
- 不改变 A3 路径或 A5 之外的架构行为。
- 不新增用户可见 CLI flag、公开 IR op 或公开 type。
- 不在本 change 中引入跨 block、跨 sync、跨 region 的通用融合框架。

## 预期结果

- 在 `--enable-op-fusion` 下，符合条件的 mixed chain 能够形成同一个 fusion group，并沿现有 OP-Lib 流水线进入 fused helper。
- inline 后的 canonical fused helper 不再停留在“多个并列 `vec_scope` stage”形态，而是收敛成单个 `pto.simd.vec_scope` 和单个 loop nest。
- 链内仅供后继 stage 消费的中间 tile 不再执行额外 `vector.maskedstore` 再 `vector.maskedload`。
- 最终输出所需的 store 仍保留，EmitC 继续能够合法地产生 `__VEC_SCOPE__`。
- 回归测试能够显式区分“helper 已生成”和“low-level fusion 真正生效”。

## 成功标准

- OpenSpec 中新增 `oplib-transforms` capability，明确 mixed chain 分组、vec-scope-aware low-level fusion 和链内中间访存消除的契约。
- OpenSpec 中对 `oplib-lowering` 的补充明确 grouped lowering path 必须与 single-op path 对上述 12 个 op 保持能力一致。
- 变更完成后，针对 `softmax_chain.pto` 一类 mixed chain，用例可以验证 fused helper 中只保留一个 `pto.simd.vec_scope`，且不再出现链内中间 tile 的 round-trip 访存。
- 变更完成后，测试可以区分合法融合、保守不融合和 EmitC 端合法代码生成三类结果。
