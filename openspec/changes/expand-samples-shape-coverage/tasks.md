## 1. Shape Profile Framework

- [ ] 1.1 在 `test/samples/runop.sh` 增加 `--shape-profile` 参数解析，支持 `default/loop4/dyn2/phase1`。
- [ ] 1.2 实现 profile 到 shape 集的统一映射：`loop4={1x32,1x96,32x32,32x96}`，`dyn2={1x32,32x96}`。
- [ ] 1.3 实现默认回退策略：未指定 profile 时保持现有 `32x32` 逻辑不变。
- [ ] 1.4 补充 `runop.sh --help` 文档与示例命令。

## 2. Phase1 Sample Parameterization (24 dirs)

- [ ] 2.1 新增 phase1 目录清单并固化 24 个 2D 目录：`Abs/Addc/Adds/Addsc/Subc/Subs/Subsc/Mul/Muls/Div/Divs/Divs2/Max/Min/Cmp/Cmps/Sel/Sels/Rowsum/Rowmax/Rowmin/Colsum/Colmax/Colmin`。
- [ ] 2.2 在上述目录样例中引入统一 shape 输入接口（默认 `32x32`，可由 runner 透传逻辑 `rows/cols`）。
- [ ] 2.3 确保参数化后 `default` profile 产物与当前行为一致（不引入功能回归）。
- [ ] 2.4 为参数化改造补充最小文档说明（入口参数、默认值、兼容性）。

## 3. Dynamic Shape Path (light 2-shape set)

- [ ] 3.1 为动态路径定义代表目录清单（至少覆盖 dynamic valid shape 与 dynamic tensor_view 两类路径）。
- [ ] 3.2 在代表目录接入 `dyn2` 两组 shape：`1x32` 与 `32x96`。
- [ ] 3.3 补齐动态路径的 golden/compare 适配，确保结果可重复验证。
- [ ] 3.4 增加动态路径冒烟回归命令并纳入本地验证脚本说明。

## 4. Shape-aware Output/Log Naming

- [ ] 4.1 为非 `default` profile 的 `.pto/.ptobc/.cpp` 与日志统一追加 `-r{rows}c{cols}` 后缀。
- [ ] 4.2 保证同一用例多 shape 产物可并存，不会相互覆盖。
- [ ] 4.3 在 summary 输出中显示 shape 维度，便于失败定位。

## 5. Loop Coverage Oracles on generated C++

- [ ] 5.1 新增 `*-pto.cpp` 循环覆盖检查逻辑，以 codegen 结果作为验收口径。
- [ ] 5.2 在 `loop4` profile 下校验四种组合：外1内1、外1内多、外多内1、外多内多。
- [ ] 5.3 在 runner 失败摘要中输出缺失组合类型与对应 shape。
- [ ] 5.4 增加一组针对 oracle 的回归测试（至少含 elementwise 与 reduction 各一例）。

## 6. CI Integration (PR lightweight + nightly full)

- [ ] 6.1 在 CI 中新增 PR 轻量 shape 任务（哨兵目录集），保持主门禁反馈时延可控。
- [ ] 6.2 在夜间/手动 workflow 新增 phase1 全量 shape 矩阵任务。
- [ ] 6.3 明确 Phase1 不接入远端 NPU shape 矩阵门禁，并在文档中记录后续 Phase2 入口条件。
- [ ] 6.4 执行 `openspec status --change expand-samples-shape-coverage --json`、`openspec validate expand-samples-shape-coverage`、`openspec show expand-samples-shape-coverage` 并记录结果。
