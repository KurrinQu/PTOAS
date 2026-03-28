# Proposal: 扩展 `test/samples` 的 shape 覆盖（Phase1）

## 概述

为 `test/samples` 引入可配置的 shape 覆盖模式，在保持默认 `32x32` 回归行为不变的前提下，新增静态与动态 shape 矩阵化验证能力，覆盖外层/内层循环 `1次/多次` 组合。

## 背景与动机

当前大量样例固定为 `32x32`，导致循环形态覆盖不足，难以及时发现以下问题：

- 外层/内层循环边界组合缺失引发的 lowering 回归。
- 动态 valid shape 与动态 tensor_view 维度路径缺乏系统性验证。
- CI 缺少可控粒度的 shape 覆盖策略（PR 快速反馈 vs 夜间全面覆盖）。

## 目标

- 提供统一 shape profile 机制：`default` / `loop4` / `dyn2` / `phase1`。
- 静态 shape 覆盖固定 4 组逻辑 shape：`1x32`, `1x96`, `32x32`, `32x96`。
- 动态 shape 覆盖固定 2 组逻辑 shape：`1x32`, `32x96`。
- 默认 `runop.sh all` 行为保持现状（仅 `32x32`）。
- 采用 PR 轻量 + 夜间全量的 CI 分层策略。
- Phase1 先覆盖 24 个 2D 核心样例目录。

## 非目标

- 不在本 change 中扩展 1D/5D 矩阵化覆盖（沿用现有专用样例）。
- 不在本 change 中接入远端 NPU shape 矩阵门禁。
- 不改变物理 tile 容量与配置（仍保持 `32x32` 物理 tile）。
- 不引入并行 `*_shape.py` 样例副本（采用参数化原样例）。

## 预期结果

- `test/samples` 能按 profile 生成多 shape 产物，且不会覆盖已有输出文件。
- 可在生成的 `*-pto.cpp` 层面验证循环覆盖组合是否到位。
- CI 可在 PR 阶段快速守护核心样例，在夜间阶段做完整 phase1 shape 矩阵回归。

## 成功标准

- `openspec/changes/expand-samples-shape-coverage/` 完成 proposal/spec/design/tasks 并通过校验。
- `runop.sh` 在 `default` profile 下输出与现有行为等价。
- `loop4` profile 可稳定覆盖静态四种循环组合：
  - 外1内1：`1x32`
  - 外1内多：`1x96`
  - 外多内1：`32x32`
  - 外多内多：`32x96`
- `dyn2` profile 在动态路径覆盖 `1x32` 与 `32x96` 两组 shape。
- 产物命名统一包含 `-r{rows}c{cols}` 后缀（非 default）。

## Capabilities

### New Capabilities

- `samples-shape-coverage`: 为 `test/samples` 提供 profile 化 shape 覆盖、shape 感知产物命名、循环覆盖验收、CI 分层门禁策略。

### Modified Capabilities

- 无

## Impact

- 主要影响：
  - `test/samples/runop.sh`
  - Phase1 样例脚本（24 个 2D 目录，参数化改造）
  - shape 覆盖配置/清单文件（新增）
  - CI workflow（PR 轻量任务 + 夜间全量任务）
- 对外行为：
  - 新增 `runop.sh` shape profile 用法。
  - 默认执行路径保持兼容，不改变现有开发者习惯。
