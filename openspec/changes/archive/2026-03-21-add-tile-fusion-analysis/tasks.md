## 1. OpenSpec 契约

- [x] 1.1 新增 `openspec/changes/add-tile-fusion-analysis/specs/tile-fusion-analysis/spec.md`，定义 `PreFusionAnalysisPass` 的输入、输出和边界语义。
- [x] 1.2 在 `proposal.md` 和 `design.md` 中明确本 change 只覆盖 5.1，不包含分组和调度。

## 2. 分析实现

- [x] 2.1 在 `include/PTO/Transforms/Passes.h`、`include/PTO/Transforms/Passes.td` 和 `lib/PTO/Transforms/` 中声明并实现 `PreFusionAnalysisPass`。
- [x] 2.2 提取统一的 `FusionOpSemantics` 帮助层，支持 SSA / DPS 双输入归一化。
- [x] 2.3 为当前 planning 范围内的 compute family 建立 block-local DFG、生命周期和迭代域分类。
- [x] 2.4 将 `treshape` 建模为局部非穿透边界：经过它的依赖链不穿透，与其无关的其他 op 不被整体阻断。
- [x] 2.5 为动态 shape 不可证场景输出稳定的保守结果，而不是隐式猜测相等。

## 3. 可测试性与验证

- [x] 3.1 增加 analysis dump 或等价测试钩子，便于 lit 观察 `PreFusionAnalysisPass` 的结果。
- [x] 3.2 增加 SSA 输入回归，验证 DFG / 生命周期 / 迭代域输出。
- [x] 3.3 增加 DPS 输入回归，验证与 SSA 输入的分析结论一致。
- [x] 3.4 增加 `treshape` 边界回归，验证 `OPA -> treshape -> OPB` 不形成穿透融合依赖。
- [x] 3.5 增加 dynamic-shape negative 回归，验证不可证 case 保守输出。
