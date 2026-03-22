## 1. OpenSpec 契约

- [x] 1.1 新增 `openspec/changes/add-op-scheduling-pass/specs/tile-fusion-scheduling/spec.md`，定义 `OpSchedulingPass` 的聚拢和合法性边界。
- [x] 1.2 在 `proposal.md` 和 `design.md` 中明确本 change 只覆盖 5.4，不重新决定 group。

## 2. 调度实现

- [x] 2.1 在 `include/PTO/Transforms/Passes.h`、`include/PTO/Transforms/Passes.td` 和 `lib/PTO/Transforms/` 中声明并实现 `OpSchedulingPass`。
- [x] 2.2 让 `OpSchedulingPass` 直接消费 `FusionPlanPass` 输出的 `pto.fusion.group_id` / `pto.fusion.order`。
- [x] 2.3 实现 block-local 稳定拓扑压缩，将同组成员聚拢成连续片段。
- [x] 2.4 明确区分 `HardBoundary` 与 `LocalBoundary`，允许 group 跨过无关的 `treshape` 移动。
- [x] 2.5 为所有移动操作补齐 SSA 定义点、side-effect、barrier、外部 call、region / block 合法性检查。

## 3. 验证

- [x] 3.1 增加基础调度回归，验证同组成员在 block 内变为连续片段。
- [x] 3.2 增加 `treshape` 调度回归，验证 group 可跨过无关 `treshape` 聚拢。
- [x] 3.3 增加 negative 回归，验证不能跨越 SSA 定义、hard boundary 或 region / block 边界。
- [x] 3.4 使用 `test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto` 做 driver sample 验证，确认两个主热点在调度后形成连续运行片段。
