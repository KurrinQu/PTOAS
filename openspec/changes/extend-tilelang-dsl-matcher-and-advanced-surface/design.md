## Context

### 范围

本 design 覆盖两个 follow-up 能力：

1. `tilelang-dsl-kernel-matcher`
2. `tilelang-dsl-advanced-surface`

它们都建立在 core-foundation 与 authoring-vpto-lowering 两个 change 之上。  
目标不是推翻 v1，而是在 v1 已稳定的前提下，把被延期的 guide surface 升级成正式 capability。

### 前置 change 依赖

本 change 以两个已经完成的前置 change 作为硬前提：

- `add-tilelang-dsl-core-foundation`
- `add-tilelang-dsl-authoring-vpto-lowering`

其中前者提供 `tilelang-dsl-surface` 与 `tilelang-dsl-diagnostics` capability，后者提供 `tilelang-dsl-vpto-lowering` capability。  
本 change 的 matcher 与 advanced surface 设计 MUST 复用这些已固定的 package/layout、descriptor API、frontend diagnostics 和 authoring-form VPTO legality contract，而不是重新发明另一套基础接口或 IR 收口。

### 当前状态

当前 v1 规划已经明确：

- 只接受单一 monomorphic `dtypes`
- 只接受显式 `strict_vecscope`
- 只支持固定 elementwise 套餐

因此，以下 guide 能力仍处于“文档存在、实现未承诺”的状态：

- 多 kernel / constraint / priority matcher
- `Any*` / `TypeVar`
- implicit vecscope inference
- raw pointer / low-level DMA authoring
- compare/select、predicate movement、carry、rearrangement 等 advanced family，以及仍待后续收口的 reduction 方向

如果不把这些能力定义成单独的 follow-up change，v1 diagnostics 将长期缺少正式的迁移目标。

### 实现约束

- 继续保持 `tilelang-dsl/` 是本特性的源码与测试承载目录。
- matcher 和 advanced surface 都必须最终收敛到当前 repo 的 authoring-form VPTO legality contract。
- 显式 `strict_vecscope` 仍是强边界，advanced inference 不能破坏这一点。
- 新 API 必须 deterministic，不能让“同样的 kernel 集合、同样的输入”出现不稳定选择结果。

## Goals / Non-Goals

**Goals:**

- 定义 kernel registry / selection 的明确接口和匹配顺序。
- 让 `constraints`、`priority`、多 signature `dtypes`、`Any*`、`TypeVar` 进入正式契约。
- 让 implicit vecscope inference、raw pointer surface、low-level DMA 和 advanced family 进入正式 capability。

**Non-Goals:**

- 不改变 v1 核心 capability 中已经定义的 package/目录边界。
- 不支持 `a5` 之外的 target。
- 不把 TileLang DSL 改造成任意 Python 语法执行器。

## Decisions

### 1. matcher 采用显式 registry + selection API，而不是隐式扫描所有 Python function

决策：

- `@pto.vkernel` 定义的 descriptor 自动注册到 module-level `KernelRegistry`
- 公开 `pto.select_kernel(target, op, operand_types, context_attrs, registry=None)` 作为 selection 入口
- 选中的 descriptor 继续复用 v1 的 `specialize()` / `mlir_text()` / `verify()` 流程
- 默认 registry 必须是显式对象，外部调用方 MAY 传入自定义 registry 做隔离测试或局部覆盖；实现 MUST NOT 通过扫描 Python globals/locals 隐式发现候选 kernel

原因：

- registry + selection API 比“由外部框架自己 introspect Python globals”更稳定、可测试。
- 这让 matcher capability 能独立存在，而不强绑某个上游 compiler integration。

### 2. selection 顺序固定为 target -> op -> dtype signature -> constraints -> priority -> tie error

决策：

- 先按 `target`
- 再按 `op`
- 再按 `dtypes` signature 做 concrete / wildcard / type-variable 匹配
- 再评估 `constraints`
- 剩余候选按最高 `priority` 选择
- 若最高 `priority` 仍有多个候选，则报 deterministic tie error，不做隐式 tiebreak
- 对单个 descriptor 内的多个 signature，匹配必须逐个 signature 独立求值；`TypeVar` 只在当前 signature 内绑定，不跨 signature 共用状态

原因：

- 这与 guide 中的叙述一致，同时避免“靠定义顺序兜底”的隐式行为。

### 3. `Any*` 与 `TypeVar` 进入 matcher capability，而不是回写 v1 surface

决策：

- `AnyFloat`、`AnyInt`、`AnyType`、`AnyMask` 只在 matcher capability 中生效
- `TypeVar("T")` 只用于单个 signature 内的位置一致性约束
- v1 核心 surface 不回溯修改；advanced capability 启用后再开放这些写法

原因：

- 这样可以保持 v1 core 仍然简单，同时让 follow-up change 独立定义 wildcard/type-variable 语义。

### 4. implicit vecscope inference 作为 advanced surface 的默认行为，但 `strict_vecscope` 继续是硬边界

决策：

- 当用户在 advanced mode 下省略显式 scope，并书写连续的 supported vector chain 时，frontend 默认推断 `pto.vecscope`
- scalar op、控制流边界、外部 call、以及显式 `strict_vecscope` 都会切断 inference
- `strict_vecscope` 继续保留，且 inference MUST NOT 穿越其边界
- inference 产物仍必须收敛到与 v1 lowering 相同的 authoring-form VPTO legality contract，不能因为自动分组而放宽 typed-mask、地址形态或 capture 规则

原因：

- 这与 guide 的默认 authoring 体验一致。
- 同时保留 `strict_vecscope` 作为 deterministic 边界，避免 inference 影响关键 kernel 的资源边界。

### 5. advanced surface 扩展 raw pointer / low-level DMA / advanced family，但继续收敛到 authoring-form VPTO

决策：

- raw pointer / low-level DMA surface 增加：
  - `castptr`
  - `addptr`
  - raw UBRef load/store
  - low-level DMA programming
  - `copy_ubuf_to_ubuf`
- advanced vector family 增加：
  - compare/select
  - predicate movement
  - carry family
  - rearrangement

- reduction family 暂不纳入本 change 的实现闭环；在 repo 暴露可直接 authoring 的公开 VPTO op 之前，frontend 继续显式 reject 这组 surface

这些 surface 仍必须 lower 到当前真实的 authoring-form VPTO，而不是发明新的公开中间 IR。
对尚未列入 capability set 的 family，frontend 继续保持 fail-fast reject，不因 advanced mode 打开而默认放开全部 surface。

原因：

- compare/select、predicate movement、carry、rearrangement 目前都能在 repo 中找到稳定的 authoring-form VPTO op，可直接作为 lowering 收口。
- reduction 目前只有 OpLib / EmitC 相关路径，没有可供 TileLang DSL 直接落地的公开 authoring-form VPTO op；如果继续把它写进当前 capability，只会迫使 frontend 发明额外中间 IR，违反本 change 的收口原则。

## Risks / Trade-offs

- [Risk] matcher capability 引入 registry 和 selection API，会让 package surface 明显扩大  
  Mitigation：把 registry/query API 单独收敛在 matcher capability，避免污染 v1 core descriptor API。

- [Risk] implicit vecscope inference 可能让 scope boundary 难以调试  
  Mitigation：保留 `strict_vecscope` 作为显式硬边界，并要求 inference 在 control-flow / scalar boundary 上切断。

- [Risk] advanced family 范围过宽，容易再次失控  
  Mitigation：按 capability 明确列出 family 分组，并用 regression 锁定首批支持面，其他 family 继续 reject。

- [Risk] raw pointer / low-level DMA authoring 可能让用户绕过高层安全网  
  Mitigation：advanced surface 继续要求最终输出通过同一套 VPTO legality contract，不因“更底层”而放宽最终收口。
