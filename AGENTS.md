# Repository Guidelines

## Scope
- 本文件适用于当前仓库根目录及其所有子目录。
- 用户直接指令优先；本文件用于补充仓库默认约定。
- 修改前先读现有实现和相邻测试，不要基于旧设计记忆直接下判断。

## Project Overview
- `PTOAS` 是基于 `LLVM/MLIR llvmorg-19.1.7` 的 out-of-tree 编译工具链，使用 `C++17`。
- 主要产物：
  - `build/tools/ptoas/ptoas`：PTO IR 解析、pass pipeline、backend lowering、EmitC/A5VM 相关输出。
  - `build/tools/ptobc/ptobc`：PTO bytecode 编解码与回归工具。
  - `python/` + `install/`：MLIR Python bindings 与 PTO Python 包。

## Repository Map
- `include/PTO/`
  - Dialect、类型、Pass 声明、TableGen 定义。
- `lib/PTO/IR/`
  - PTO/A5VM dialect 实现。
- `lib/PTO/Transforms/`
  - lowering、优化、同步、代码生成主逻辑。
- `lib/PTO/Transforms/TileFusion/`
  - tile fusion 分析、planning、scheduling、region materialization、A5VM 后处理。
- `tools/ptoas/`
  - `ptoas` CLI 入口和主 pipeline 拼装。
- `tools/ptobc/`
  - `ptobc` 源码、测试与维护脚本。
- `oplib/`
  - OP-Lib 模板、family 资源、实例化输入。
- `python/`, `lib/Bindings/Python/`, `lib/CAPI/`
  - Python binding 与 C API。
- `test/basic`, `test/oplib`, `test/tile_fusion`, `test/phase2`, `test/phase3`, `test/npu_validation`, `test/samples`
  - lit 回归、OP-Lib 回归、tile fusion 回归、端到端样例与 NPU 验证辅助。
- `docs/`, `docs/tile_fusion/`
  - 设计文档、用户文档、阶段性方案说明。
- `openspec/`
  - 行为契约、变更提案、设计与任务分解。
- `build/`, `install/`, `output/`, `output_log/`
  - 生成产物。除非任务明确要求，否则不要把它们当作 source of truth。

## Environment And Build
- 仓库当前实际使用的环境脚本是 `scripts/ptoas_env.sh`，不是旧的 `env.sh` / `do_cmake.sh`。
- 在运行 `ptoas`、`ptobc`、Python binding、sample 或 lit 之前，优先：

```bash
source scripts/ptoas_env.sh
```

- 默认目录约定由该脚本推导：
  - `PTO_SOURCE_DIR=<workspace>/PTOAS`
  - `LLVM_BUILD_DIR=<workspace>/llvm-project/build-shared`
  - `PTO_INSTALL_DIR=<repo>/install`
- 典型配置/构建流程：

```bash
export PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)

cmake -G Ninja -S . -B build \
  -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
  -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
  -DPython3_EXECUTABLE=$(which python3) \
  -DPython3_FIND_STRATEGY=LOCATION \
  -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_PYTHON_PACKAGE_DIR=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core \
  -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR"

ninja -C build
ninja -C build install
```

## A5 Semantic Source Of Truth
- 修改 A5 lowering、LLVM emission、sample 语义或 validation oracle 时，先查看 `ASCEND_HOME_PATH` 下已安装的 PTO 实现，把它当作语义基线。
- 不要仅根据仓库内 lowering / emitter 代码反推 A5 语义；只有在已安装 PTO 头文件和真实工具链产物都确认一致时，才允许替换 intrinsic / wrapper 关系。
- 如果头文件信息不够，优先通过当前 Bisheng 工具链结合 testcase 构建参数、`-v`、`-save-temps` 等方式追踪真实产物，再改行为。

## Commit & Pull Request Guidelines
- commit subject 使用简短祈使句，例如 `Fix explicit StringAttr bool conversion`。
- 单个 commit 保持一个清晰主题。PR 描述要写清受影响 lowering 路径、实际跑过的验证命令、新增通过或有意延期的 sample，以及必要的 IR / 输出片段。

- 常用二进制位置：
  - `build/tools/ptoas/ptoas`
  - `build/tools/ptobc/ptobc`

## Coding Conventions
- 遵循 LLVM/MLIR 附近文件的风格，不要在本仓库强行套用另一套格式。
- TableGen 使用 2 空格缩进；C++ 保持与相邻文件一致。
- 默认使用 ASCII；只有目标文件已使用 Unicode 时才跟随。
- 新 helper / utility 命名优先描述行为和边界，例如 `build...Contract`、`lower...`、`extract...`。
- 新 pass 应保持 LLVM/MLIR 常规接线方式：
  - 在 `include/PTO/Transforms/Passes.h` 声明；
  - 在 `include/PTO/Transforms/Passes.td` 注册选项和构造器；
  - 在 `lib/PTO/Transforms/` 或 `lib/PTO/Transforms/TileFusion/` 实现；
  - 需要 CLI 可达时，再接到 `tools/ptoas/ptoas.cpp`。

## Testing And Validation
- 先跑最小相关验证，再扩大范围；不要默认全量重编译或全量回归。
- `test/lit.cfg.py` 主要发现 `.mlir` 测试。pass/lowering 变更优先用 lit。
- 常用验证命令：

```bash
python3 -m lit -sv test/basic/<target>.mlir
python3 -m lit -sv test/oplib/<target>.mlir
python3 -m lit -sv test/tile_fusion/<target>.mlir
python3 -m lit -sv test/tile_fusion
bash test/samples/runop.sh -t Abs
bash test/samples/runop.sh -t TileFusion
bash test/samples/runop.sh all
bash test/phase3/run_phase3_checks.sh
```

- `runop.sh` 支持这些常用覆盖变量：
  - `PTOAS_BIN`
  - `PTOBC_BIN`
  - `PYTHON_BIN`
  - `PTOAS_FLAGS`
  - `PTOAS_ENABLE_INSERT_SYNC`
  - `PTOAS_OUT_DIR`
  - `--enablebc`
- 如果修改 `ptobc`、opcode 映射、PTO op schema 或 bytecode 兼容性，至少补跑：

```bash
ctest --test-dir build -R ptobc_stage9_e2e
ctest --test-dir build -R ptobc_to_ptoas_smoke
ctest --test-dir build -R ptobc_opcode_coverage_check
```

- 如果变更会影响板端/NPU 路径，额外留意：
  - `test/npu_validation/`
  - `scripts/run_output_npu_validation.sh`
  - `scripts/batch_generate_output_npu_testcases.sh`

## Tile Fusion And A5VM Baseline
- 当前 tile fusion 主实现集中在：
  - `lib/PTO/Transforms/TileFusion/PTOPreFusionAnalysis.cpp`
  - `lib/PTO/Transforms/TileFusion/PTOFusionPlan.cpp`
  - `lib/PTO/Transforms/TileFusion/PTOOpScheduling.cpp`
  - `lib/PTO/Transforms/TileFusion/PTOFusionRegionGen.cpp`
  - `lib/PTO/Transforms/TileFusion/PTOLowLevelLoopFusion.cpp`
  - `lib/PTO/Transforms/TileFusion/PTOFusionLoadStoreElision.cpp`
  - `lib/PTO/Transforms/TileFusion/PTOFlattenFusionRegion.cpp`
- `tools/ptoas/ptoas.cpp` 当前显式拼装的 A5 fusion 主线是：
  - `FusionPlan -> OpScheduling -> PTOFusionRegionGen -> PTOToA5VM -> PTOLowLevelLoopFusion -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion`
- backend 相关变更优先验证 `--pto-backend=a5vm --a5vm-print-ir`，先看 A5VM IR，再看文本/LLVM 发射结果。
- 涉及 tile fusion 合法性、region 边界、grouped lowering 或 store elision 时，不要只看单个 pass；至少同时检查：
  - 相邻 pass 的输入/输出契约；
  - `tools/ptoas/ptoas.cpp` 中的 pipeline 顺序；
  - 对应 `test/tile_fusion/` 回归；
  - 相关 `openspec/specs/` 契约。

## OpenSpec Workflow
- 本仓库已启用 `openspec/` 工作流；涉及外部可观察行为、lowering 契约、OP-Lib capability、tile fusion 阶段边界时，优先核对 OpenSpec。
- 当前常见基线 spec 包括：
  - `openspec/specs/tile-fusion-analysis/spec.md`
  - `openspec/specs/tile-fusion-planning/spec.md`
  - `openspec/specs/tile-fusion-scheduling/spec.md`
  - `openspec/specs/tile-fusion-region-encapsulation/spec.md`
  - `openspec/specs/tile-fusion-region-lowering/spec.md`
  - `openspec/specs/tile-fusion-store-elision/spec.md`
  - `openspec/specs/a5vm-backend-pipeline/spec.md`
  - `openspec/specs/oplib-lowering/spec.md`
- `openspec/config.yaml` 约定 OpenSpec 产物默认使用简体中文。编写 proposal/design/tasks 时，保留代码符号和路径原文，不翻译 op 名、pass 名、命令行参数。
- 小型 bugfix、局部重构、纯文档更新通常不需要新增 proposal；但如果更改用户可见语义或阶段契约，应先确认是否已有对应 change/spec。

## Change Boundaries
- 优先修改源文件：
  - `include/`
  - `lib/`
  - `tools/`
  - `python/`
  - `oplib/`
  - `test/`
  - `docs/`
  - `openspec/`
- 避免手改生成目录：
  - `build/`
  - `install/`
  - `output/`
  - `output_log/`
- `test/samples/` 下不少 `.cpp` / 中间输出来自脚本或 `.pto` 生成；能重生就不要手修生成物。
- 如果改动 CLI 选项、默认路径、backend 选择逻辑，要同时检查 sample 脚本和相关文档是否也要同步。

## Practical Defaults
- 搜索优先用 `rg`。
- 先读实现，再读相邻测试，再修改。
- 任何涉及 A3/A5、EmitC/A5VM、single-op/grouped lowering 分叉的变更，都先确认当前代码是否已经分流，不要假设一条路径覆盖所有架构。
- 无法运行验证时，明确说明是缺少 LLVM 构建、Python 环境、NPU 依赖，还是目标二进制未编译。
