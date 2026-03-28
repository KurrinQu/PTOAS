## Context

当前 `test/samples` 以 `32x32` 为主，循环形态与动态 shape 覆盖不足。现有 CI 主门禁直接运行 `runop.sh --enablebc all`，如果无分层策略地扩展 shape，容易显著拉长 PR 反馈时间。
本设计在不破坏默认行为的前提下，引入 profile 化 shape 覆盖能力，优先解决 Phase1 的 2D 核心样例覆盖。

## Goals / Non-Goals

**Goals:**

- 为 `runop.sh` 提供统一 shape profile 调度能力：`default/loop4/dyn2/phase1`。
- 用逻辑 shape 覆盖外层/内层循环 `1次/多次` 四种组合（静态）与动态路径轻量覆盖。
- 通过 shape 感知命名确保多 shape 产物与日志可并存。
- 按 PR 轻量 + 夜间全量方式接入 CI。
- Phase1 完成 24 个 2D 样例目录参数化改造。

**Non-Goals:**

- 不引入并行 `*_shape.py` 样例副本。
- 不在本阶段扩展 1D/5D shape 矩阵覆盖。
- 不在本阶段接入远端 NPU 的 shape 矩阵门禁。
- 不改变物理 tile `32x32` 的容量与硬件约束。

## Decisions

### 决策 1：参数化原样例，而不是新增并行样例

- 选择：在现有样例脚本中增加统一 shape 参数入口（环境变量或 runner 透传参数）。
- 理由：避免脚本重复维护与 golden/compare 漂移。
- 备选方案：
  - 新增 `*_shape.py`：短期实现快，但长期维护成本高，易与主样例行为分叉。

### 决策 2：profile 化执行模型

- 选择：引入 `default/loop4/dyn2/phase1` 四类 profile。
- 理由：兼容默认行为并可分层扩展覆盖。
- 备选方案：
  - 直接把 `all` 改为多 shape：会显著增加 PR 门禁成本，反馈变慢。

### 决策 3：逻辑 shape 覆盖循环组合，物理 tile 固定

- 选择：静态 shape 集固定为 `1x32`, `1x96`, `32x32`, `32x96`；动态轻量集固定为 `1x32`, `32x96`。
- 理由：精准覆盖循环组合，同时遵循现有 tile 约束。
- 备选方案：
  - 引入更多 tail/big shape：覆盖更强但 Phase1 风险和成本偏高。

### 决策 4：Phase1 目录清单固定化

- 选择：先覆盖 24 个 2D 目录：
  - `Abs`, `Addc`, `Adds`, `Addsc`, `Subc`, `Subs`, `Subsc`, `Mul`, `Muls`, `Div`, `Divs`, `Divs2`, `Max`, `Min`, `Cmp`, `Cmps`, `Sel`, `Sels`, `Rowsum`, `Rowmax`, `Rowmin`, `Colsum`, `Colmax`, `Colmin`
- 理由：覆盖 elementwise + compare + reduce 主路径，复杂度可控。
- 备选方案：
  - 一次性全量目录：落地周期长，回归失败定位复杂。

### 决策 5：循环覆盖验收口径

- 选择：以生成的 `*-pto.cpp` 为验收口径，验证外/内循环 `1次/多次` 组合。
- 理由：直接对应最终 codegen 结果，便于回归定位。
- 备选方案：
  - 以 IR 为准：更早阶段可见，但与最终 codegen 行为可能存在偏差。

## Risks / Trade-offs

- [Risk] 样例脚本参数化后出现行为漂移 -> 通过 `default` profile 回归对齐现有输出并新增哨兵样例比对。
- [Risk] 多 shape 产物与日志命名不一致导致排障困难 -> 统一后缀规范 `-r{rows}c{cols}` 并在 runner 中集中生成。
- [Risk] CI 耗时上升影响 PR 体验 -> 采用 PR 轻量集 + 夜间全量集分层策略。
- [Risk] 动态路径覆盖不足导致 false confidence -> 在 Phase1 明确动态 valid shape + 动态 tensor_view 两条路径均需覆盖。

## Migration Plan

1. 新增 shape profile 框架与配置清单，不改变默认执行路径。
2. 先改造 24 个 phase1 目录样例脚本，完成静态 `loop4` 覆盖。
3. 在代表目录接入 `dyn2` 动态覆盖并补齐必要 golden/compare 适配。
4. 落地 shape 感知产物命名与日志命名。
5. 增加 `*-pto.cpp` 循环覆盖检查脚本并挂接到 runner。
6. CI 接入 PR 轻量任务与夜间全量任务，观测稳定后再评估 Phase2（远端 NPU 接入）。

## Open Questions

- Phase2 是否纳入 1D/5D 矩阵化覆盖。
- Phase2 是否将 shape 矩阵接入远端 NPU 验证门禁。
