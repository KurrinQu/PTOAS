## ADDED Requirements

### Requirement: Shape Profile Selection

`test/samples/runop.sh` SHALL 支持通过 profile 选择 shape 覆盖模式，且默认行为必须保持与现有 `32x32` 回归一致。

#### Scenario: 选择 profile 并执行对应 shape 集

- **WHEN** 用户执行 `runop.sh` 并指定 `--shape-profile` 为 `default`, `loop4`, `dyn2` 或 `phase1`
- **THEN** runner 必须按所选 profile 生成并执行对应 shape 集，且 `default` 必须仅运行既有 `32x32` 行为

### Requirement: Static Loop Combination Coverage

系统 SHALL 通过静态逻辑 shape 集覆盖外层/内层循环 `1次/多次` 的四种组合，并保持物理 tile 为 `32x32` 不变。

#### Scenario: 使用 `loop4` 覆盖四种循环组合

- **WHEN** 用户选择 `loop4` profile
- **THEN** 系统必须覆盖逻辑 shape `1x32`, `1x96`, `32x32`, `32x96`，分别对应外1内1、外1内多、外多内1、外多内多

### Requirement: Dynamic Shape Coverage (Lightweight)

系统 SHALL 在 Phase1 中以轻量模式覆盖动态 shape 路径，至少包含动态 valid shape 与动态 tensor_view 维度两类路径。

#### Scenario: 使用 `dyn2` 覆盖动态路径

- **WHEN** 用户选择 `dyn2` profile
- **THEN** 系统必须在动态路径执行逻辑 shape `1x32` 与 `32x96`，并对动态 valid shape 与动态 tensor_view 两类路径均完成至少一组覆盖

### Requirement: Shape-aware Artifact Naming

系统 SHALL 对非 default profile 产生的产物使用 shape 感知命名，避免覆盖同名文件并支持后续排查。

#### Scenario: 生成 shape 感知命名产物

- **WHEN** runner 在非 `default` profile 下生成 `.pto`、`.cpp` 与日志
- **THEN** 输出文件名必须包含 `-r{rows}c{cols}` 后缀（例如 `addc-r1c96-pto.cpp`），且不同 shape 结果必须可并存

### Requirement: CI Gating Strategy for Shape Coverage

系统 SHALL 采用 PR 轻量 + 夜间全量的 shape 覆盖门禁策略，以平衡反馈速度与覆盖率。

#### Scenario: PR 与夜间任务分工

- **WHEN** CI 在 pull request 触发执行
- **THEN** 必须仅运行哨兵样例的轻量 shape 覆盖任务；并在夜间或手动触发时运行 phase1 全量 shape 矩阵任务

### Requirement: Golden Precision Comparison

系统 SHALL 对每个 shape case 的测试结果执行与 golden 数据的精度对比，并将不满足精度阈值的 case 判定为失败。

#### Scenario: 每个 shape case 必须完成精度对比

- **WHEN** runner 在任一 profile（`default`, `loop4`, `dyn2`, `phase1`）下完成某个 case 的执行并产生输出
- **THEN** 系统必须对该 case 运行对应的 golden compare（按 dtype 使用既定容差规则），且 compare 失败时该 case 状态必须为 FAIL
