## 1. OpenSpec 契约

- [x] 1.1 新增 `openspec/changes/add-tile-fusion-region-encapsulation/specs/tile-fusion-region-encapsulation/spec.md`，定义 5.5 `PTOFusionRegionGenPass`、`pto.fusion_region` 和 `pto.yield` 的正式契约。
- [x] 1.2 在 `proposal.md` 和 `design.md` 中明确 5.5 采用 region-based 输出，并把 helper-function outline 视为旧模型而不是正式目标形态。

## 2. PTO Dialect IR 容器

- [x] 2.1 在 `include/PTO/IR/PTOOps.td` 中声明 `pto.fusion_region` 和 `pto.yield`，固定 single-block body、variadic inputs / outputs 与 terminator 约束。
- [x] 2.2 在 `lib/PTO/IR/PTO.cpp` 中补齐 verifier、parser/printer 或所需的辅助逻辑，确保 `pto.yield` 只出现在 `pto.fusion_region` 内部，且结果顺序与 region 接口一致。
- [x] 2.3 在 PTO dialect 相关头文件和 CMake 接线中注册新 op，确保 pass 与测试可直接使用新 IR。

## 3. 5.5 封装 Pass

- [x] 3.1 在 `include/PTO/Transforms/Passes.h`、`include/PTO/Transforms/Passes.td` 和 `lib/PTO/Transforms/` 中声明并实现 `PTOFusionRegionGenPass`。
- [x] 3.2 让 `PTOFusionRegionGenPass` 直接消费 `OpSchedulingPass` 输出的 `pto.fusion.group_id` / `pto.fusion.order`，按 basic block 扫描连续 group span。
- [x] 3.3 实现 external inputs / escaping outputs 提取与稳定排序规则，生成 `pto.fusion_region` 的 operands、block arguments、results 与 `pto.yield` operands。
- [x] 3.4 在封装成功后移除原组内成员上的 `pto.fusion.group_id` / `pto.fusion.order`，把 group 身份收敛到 `pto.fusion_region` 上。
- [x] 3.5 在 tile fusion pipeline 中把 5.5 接到 `OpSchedulingPass` 之后，并停止把 `PTOOutlineFusionGroups` 作为 tile-fusion 主线的目标输出契约。
- [x] 3.6 为非法输入补齐显式失败逻辑，覆盖 split span、metadata 残缺和 region 闭包失败等场景。

## 4. 验证

- [x] 4.1 增加基础 `lit` 回归，验证单个连续 fusion group 被替换为一个且仅一个 `pto.fusion_region`。
- [x] 4.2 增加多输入、多输出回归，验证 region input / output / `pto.yield` 顺序稳定且显式。
- [x] 4.3 增加 negative 回归，验证 split group、残缺 `pto.fusion.*` metadata 和隐式外部 capture 会导致 pass 失败。
- [x] 4.4 使用 `test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto` 做 focused driver sample 验证，确认主热点在 5.4 之后能被 5.5 正确封装为 region。
