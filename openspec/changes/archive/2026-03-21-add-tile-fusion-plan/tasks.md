## 1. OpenSpec 契约

- [x] 1.1 新增 `openspec/changes/add-tile-fusion-plan/specs/tile-fusion-planning/spec.md`，定义 `FusionPlanPass` 的 DAG 分组和 metadata 输出契约。
- [x] 1.2 在 `proposal.md` 和 `design.md` 中明确本 change 只覆盖 5.3，不包含调度。

## 2. 规划实现

- [x] 2.1 在 `include/PTO/Transforms/Passes.h`、`include/PTO/Transforms/Passes.td` 和 `lib/PTO/Transforms/` 中声明并实现 `FusionPlanPass`。
- [x] 2.2 建立 `StrategyEngine` / `CostModel` 内部接口，并落一个默认保守贪心规划器。
- [x] 2.3 将 planning 范围扩展到当前 driver sample 的最小闭包：12 个 binary / binary-scalar、`texp`、`texpands`、`trowexpandmul`、`trowexpanddiv`。
- [x] 2.4 在分组逻辑中落实 `treshape` 的局部非穿透语义：经过它的依赖链不成组，与其无关的其他候选组不被整体阻断。
- [x] 2.5 为每个融合组写出稳定的 `pto.fusion.group_id` 和 `pto.fusion.order`。

## 3. 验证

- [x] 3.1 增加 diamond DAG 回归，覆盖 `tmax -> tsub x2 -> texp x2 -> tmul x2 -> tadd`。
- [x] 3.2 增加 join DAG 回归，覆盖 `trowexpandmul x2 -> tadd`。
- [x] 3.3 增加 `treshape` 局部边界回归，验证依赖链不穿透、无关 op 不被全局阻断。
- [x] 3.4 增加 dynamic-shape negative 回归，验证不可证 case 保守不分组。
- [x] 3.5 使用 `test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto` 做 driver sample 验证，确认两个主热点稳定成组。
