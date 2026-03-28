## Context

### 范围

本 design 只覆盖 A5VM `vld*/vst*` 家族地址操作数契约与对应主线边界，不扩展到 `copy_* / gather / scatter` 等其他 buffer 类 op。  
覆盖对象包含：

- 无状态 vector load/store：如 `vlds/vldas/vldus/vsld/vldx2` 与 `vsts/vsst/vstx2/vsstb/vsta/vstas/vstar`
- predicate load/store 变体：如 `plds/pld/pldi/psts/pst/psti`

显式不覆盖：

- stateful store 变体 `pstu/vstu/vstus/vstur` 的 `base/base_out` memref 语义扩展

### 当前状态

- A5VM verifer 层对多类 buffer 使用了 pointer-like 语义检查，但 ODS 类型约束仍存在 pointer-only 收口，导致接口契约不统一。
- A5 backend 优化主线需要保留结构化地址语义才能稳定做 load/store 消除与后续融合优化。
- emit/LLVM 端最终依赖 pointer ABI，但当前契约没有明确“何时保留 memref、何时转 pointer”的阶段边界。

### 设计拆分

1. IR 接口层：`vld*/vst*` 双地址形态  
2. backend 主线层：`PTOToA5VM -> PTOLowLevelLoopFusion -> PTOFlattenFusionRegion` 的 memref-first 地址模型  
3. 发射边界层：A5VM text/LLVM 发射阶段完成 pointer ABI 对接  
4. stateful 边界层：`base/base_out` 继续 pointer-only

### 实现约束

- 不新增新 op 名，不引入 *_mr 分叉接口。
- 不新增 CLI flag。
- 不改变现有 A5VM intrinsic 语义与命名。
- 对未纳入范围的 `copy_* / gather / scatter` 维持现状，不借本 change 一并放宽。

### 测试策略

- IR verifier 层：
  - `vld*/vst*` 代表性 op 覆盖 memref 正例与 ptr 正例。
  - `pstu/vstu/vstus/vstur` 覆盖 memref 负例（`base/base_out` 禁止 memref）。
- pipeline 层：
  - A5 backend 主线验证 memref-first 契约可观察。
- 发射层：
  - memref 与 ptr 输入形态分别验证发射后语义等价（不出现调用行为分叉）。

## Goals / Non-Goals

**Goals:**

- 建立 `vld*/vst*` 双地址形态（memref + ptr）统一契约。
- 明确 A5 backend 主线 memref-first 阶段语义。
- 保持 ptr 形态完全兼容，满足低层直接编程诉求。
- 将 pointer ABI 收口到发射边界，不前移到优化阶段。
- 对 stateful `base/base_out` 保持 pointer-only，避免在本 change 引入未定 post-update memref 语义。

**Non-Goals:**

- 不扩展 `copy_* / gather / scatter` 等非 `vld*/vst*` buffer op。
- 不重写 A5VM 发射框架与 backend pipeline 顺序。
- 不在本 change 中定义 stateful memref `base_out` 结果类型系统。

## Decisions

### 决策 1：`vld*/vst*` 使用同名双形态，不新增新 op

- 选择：在同名 op 上同时接受 `memref` 与 `!llvm.ptr`。
- 理由：避免接口分叉；保持历史 IR 与手写 A5VM 可用；降低迁移成本。
- 备选（未采用）：新增 *_mr 专用 op。  
  原因：会把同一语义拆成两套 op，增加 pass 与文档负担。

### 决策 2：A5 backend 主线地址模型采用 memref-first

- 选择：在 `PTOToA5VM -> PTOLowLevelLoopFusion -> PTOFlattenFusionRegion` 阶段默认保留 memref 地址语义。
- 理由：优化依赖结构化地址信息；pointer-only 会提前丢失语义。
- 备选（未采用）：主线继续 pointer-only。  
  原因：会持续增加优化补偿逻辑。

### 决策 3：发射边界负责 pointer ABI 映射

- 选择：A5VM text/LLVM 发射阶段承担 memref->pointer ABI 对接。
- 理由：符合“优化阶段保留语义、发射阶段收口 ABI”的职责分离。
- 备选（未采用）：在更前段统一强制 pointer 化。  
  原因：前移会削弱中段优化能力。

### 决策 4：stateful `base/base_out` 继续 pointer-only

- 选择：`pstu/vstu/vstus/vstur` 的 `base/base_out` 不引入 memref 语义。
- 理由：`base_out` 涉及 post-update 地址语义，目前没有统一 memref view 契约；先保持稳定边界。
- 备选（未采用）：允许 stateful `base_out` 也为 memref。  
  原因：会引入新的语义设计与验证复杂度，超出本 change 范围。

## Risks / Trade-offs

- [Risk] 双形态接口导致 verifier/ODS 契约不一致  
  Mitigation：以 OpenSpec 明确“哪些 op 双形态、哪些 op pointer-only”，并配套正负例。

- [Risk] 发射边界若未完成 pointer ABI 收口，可能出现 memref 调用签名漂移  
  Mitigation：增加 memref/ptr 对照发射用例，要求调用语义等价。

- [Risk] 后续需求可能要求 stateful 也支持 memref `base_out`  
  Mitigation：本 change 明确该部分是显式非目标，后续单独 proposal 处理，避免本次边界失焦。
