## Context

### 范围

本 design 只覆盖 PTO IR manual 4.5~4.9 范围内、当前已经接入 A5 OpLib V1 的 family，以及与同级 `pto-isa` A5 语义源的对齐问题。

### 当前状态

当前 PTOAS 已具备以下基础能力：

- `oplib/level3` concrete template、Family DSL 和 manifest snapshot。
- `PTOLowerToOpLibCalls` 基于 manifest + template registry 做 mandatory lowering。
- `PTOToEmitC` 对部分未进入 OpLib 的 OP 仍保留直接或 decomposition 式 EmitC lowering。

本次 review 已确认三类偏差：

1. manifest 失真

- `trecip` 已经存在 `pto-isa` 公共 API 和 A5 ST 用例，但 manifest 仍标记为 `deferred`。
- bitwise 一组的 `dtype_support` 存在过宽记录，和 A5 头文件真实约束不一致。

2. concrete template 覆盖收缩

- compare/select 目前基本是 `f32-only`。
- row/col reduction、broadcast、scalar-expand 目前只覆盖 `f32` 或极少数子集。
- arithmetic / tile-scalar / bitwise family 对 `bf16`、`u8/u16/u32`、部分 `f16`/`i16` 也存在明显缺口。

3. 语义边界不清

- `taddc`、`tsubc`、`taddsc`、`tsubsc` 在 `pto-isa` 公共 API 中存在入口，但缺少 A5 `_IMPL`。
- PTOAS EmitC 已经通过 decomposition 维持了行为，但 OpLib V1 还没有明确是否允许用相同语义进入模板体系。

### 约束

- 不改变 `--op-lib-dir` 输入模型，lowering 仍消费 checked-in concrete `.mlir`。
- 不把同级 `pto-isa` 直接编译进 PTOAS；对齐只发生在 manifest 生成、校验和回归层。
- 不扩大到 A5 以外架构。

## Goals / Non-Goals

**Goals:**

- 让 `a5_oplib_v1_manifest.yaml` 与同级 `pto-isa` 当前 A5 语义保持一致，至少不再把已可用语义误标为 `deferred`。
- 明确 A5 OpLib V1 的“语义证据”定义：不仅包括 `include/pto/npu/a5/*_IMPL`，也包括 `pto/common/pto_instr.hpp` 的公共 API 映射、A5 ST 测试目录和可接受的 decomposition 语义。
- 为 `oplib/level3` 建立 family 级 dtype/variant 覆盖要求，优先覆盖本次 review 已确认的 compare/select、reduction、broadcast、scalar-expand、bitwise 和 arithmetic gap。
- 建立 manifest、template、lowering 三层一致性的验证路径，避免以后再靠人工 review 才发现漂移。

**Non-Goals:**

- 不在本 change 中引入新的 MLIR dialect 或新的 matcher key。
- 不要求一次性重做全部 Family DSL 生成逻辑。
- 不要求 PTOAS 在本 change 中覆盖 `pto-isa` 仓内所有非 V1 范围 API。

## Decisions

### 决策 1：把 A5 语义证据从“仅 `_IMPL` 头文件”扩展为“公共 API + A5 头文件 + A5 ST 用例”

当前 manifest 误判 `trecip` 的根因，是生成逻辑只把 `OP_NAME(...)` / `*_IMPL` 视为语义来源，漏掉了 `pto/common/pto_instr.hpp` 里的 A5 公共 API 映射。

本 change 采用如下证据优先级：

1. A5 专有头文件中的 `*_IMPL` / `OP_NAME(...)`
2. `pto/common/pto_instr.hpp` 中的公共 API 映射
3. `tests/npu/a5/src/st/testcase/*` 中的 A5 ST 用例

采用该策略的原因：

- 它能解释 `trecip` 这类“无 A5 独立 `_IMPL`，但公共 API 明确可用”的情况。
- 它不要求 PTOAS 改成解析 `pto-isa` 全量实现，只是扩大“可承认的语义证据”。

备选方案：

- 仅依赖 A5 `_IMPL`：实现简单，但会继续误伤 `trecip` 这类公共 API 复用路径。
- 完整解析 `pto-isa` 模板实例：精度更高，但复杂度超出本 change 范围。

### 决策 2：将 gap 分为“manifest 纠偏”、“模板补齐”、“保留 deferred”三类处理

不是所有 gap 都应该用同一种方式修复。

分类规则如下：

- manifest 纠偏：
  上游 A5 语义已存在，PTOAS 只是快照记录错误，例如 `trecip`、bitwise dtype。
- 模板补齐：
  上游 A5 语义已存在，PTOAS lowering 接口也能表达，但 concrete template 尚未覆盖，例如 compare/select、reduction、broadcast、scalar-expand 及部分 arithmetic/bitwise dtype。
- 保留 deferred：
  上游公共 API 虽存在，但 A5 原生 `_IMPL` 为空，且 PTOAS 尚未明确允许 decomposition 进入 OpLib family，例如 `taddc`、`tsubc`、`taddsc`、`tsubsc`。

采用该分类的原因：

- 可以避免把所有问题都错误归因成“模板没写”。
- 可以让任务拆分更清晰，先修高收益、低风险的 manifest 和模板问题。

### 决策 3：对 `trecip` 允许按公共 API 语义进入 OpLib 对齐范围

`trecip` 在 `pto-isa` 公共 API 中已被定义为 `TDIVS(dst, 1, src)` 语义，PTOAS EmitC 也已有 `TRECIP` lowering。

因此本 change 将 `trecip` 视为：

- 对 manifest：应标记为 implemented，而不是 deferred。
- 对模板：允许用与该公共 API 语义等价的 OpLib 模板或 template 选择策略进入 V1。

备选方案：

- 继续 deferred，等待 A5 独立 `_IMPL`：最保守，但与现状不符，也会让 manifest 持续失真。

### 决策 4：继续将 ternary 四个 OP 保持为单独决策点，不默认因为公共 API 存在就转 implemented

`taddc`、`tsubc`、`taddsc`、`tsubsc` 与 `trecip` 的区别在于：PTOAS 当前 EmitC decomposition 是可工作的，但上游 A5 公共 API 仍调用缺失的 `_IMPL`，没有像 `trecip` 那样的公共语义重写。

因此本 change 不强行把 ternary 四个 OP 转为 implemented，而是要求：

- manifest 明确记录为“缺少 A5 `_IMPL`，暂不进入 OpLib implemented 集”。
- 如果未来要按 decomposition 语义进入 OpLib，必须作为显式决策补文档、模板和回归，而不是在本 change 中顺带放开。

### 决策 5：模板覆盖优先按 family 补齐，而不是逐个 OP 手写散点修补

本 change 仍沿用 Family DSL + generator + concrete `.mlir` 的路线，优先在 family 级展开 dtype/variant 轴：

- compare/select：补 `f16/i16/i32/i8/u16/u32/u8` 覆盖。
- row-reduce：至少补 `f16`。
- col-reduce、broadcast、scalar-expand：按 A5 头文件真实类型矩阵展开。
- arithmetic / tile-scalar / bitwise：补 manifest 已确认且上游真实支持的 `bf16`、unsigned、部分 i16/i8 组合。

采用该策略的原因：

- 与当前 generator 维护方式兼容。
- 能减少模板数量扩张时的重复体。

### 决策 6：把一致性检查扩展到“manifest × template × lowering use case”

现有 `check_implemented_op_alignment.py` 只能检查 implemented op 是否“至少有一个模板 + 至少一个 lowering 用例”，无法发现“dtype/variant 只覆盖子集”的问题。

本 change 将把验证升级为：

- manifest implemented dtype 集必须能在 concrete template 中找到对应覆盖，允许按 family 规则声明例外。
- manifest declared dtype 不得超出同级 `pto-isa` A5 真实约束。
- 关键 family 的 lit / smoke 回归要覆盖新增 dtype/variant。

## Risks / Trade-offs

- [Risk] manifest 生成逻辑过度依赖同级 `pto-isa` 目录结构，导致路径耦合增强
  → Mitigation：只依赖有限且稳定的入口路径，例如 `include/pto/common/pto_instr.hpp`、`include/pto/npu/a5/`、`tests/npu/a5/src/st/testcase/`，并在工具中提供缺失目录的确定性报错。

- [Risk] 一次性补太多 dtype/variant，导致模板数量和回归规模快速膨胀
  → Mitigation：优先补本次已确认的高优先级 family，保持 family 级 generator 生成，不做散点手写模板。

- [Risk] `trecip` 与 ternary 家族使用不同对齐策略，容易让规则看起来不统一
  → Mitigation：在 spec 中明确“是否存在公共 API 级语义重写”是分类标准；`trecip` 有，ternary 四个没有。

- [Risk] 某些 manifest 记录可能不只是 dtype 错误，还混入无关文本或脏快照
  → Mitigation：将 manifest 校验纳入实现任务，至少覆盖 `dtype_support`、`key_constraints` 和 `deferred_reason` 的结构化检查。

## Migration Plan

1. 先更新 proposal/spec 定义的语义边界。
2. 再修 manifest 生成/校验逻辑，确保 `trecip` 和 bitwise dtype 等问题先回到正确状态。
3. 然后按 family 扩展模板和 lowering 回归。
4. 最后收紧一致性检查，避免旧的失真再次回流。

本 change 不涉及对外 CLI 迁移，也不需要用户侧 rollback 机制；若实施中发现 family 覆盖过宽，可先回退新增模板与对应 manifest 条目，再保持较窄 implemented 集。

## Open Questions

- `taddc`、`tsubc`、`taddsc`、`tsubsc` 后续是否接受“按 PTOAS decomposition 语义 implemented”的策略，还是坚持等待上游 A5 `_IMPL`？
- compare/select 的模板扩展是否全部进入 V1，还是先优先 `f16/i32/u8` 等 lit / sample 已有需求更强的子集？
- manifest 是否需要从 YAML/JSON 快照进一步提升为“可重建的生成产物 + 校验摘要”，减少手工维护风险？
