# vpto-ir-legality Specification

## Purpose
TBD - created by archiving change add-dual-stage-vpto-legality-verifier. Update Purpose after archive.
## Requirements
### Requirement: VPTO backend MUST validate authoring-form VPTO IR before emission-boundary canonicalization

在 `--pto-backend=vpto` 路径中，backend mainline 结束后 MUST 先验证 authoring-form VPTO IR，再进入 `PTOVPTOPtrBoundary`。  
该阶段的输入契约是 post-mainline VPTO IR：允许 memref-first 地址表达，也允许 hand-written ptr-form VPTO IR，但尚未进入 emission-boundary ptr canonicalization。

#### Scenario: lowered-from-PTO VPTO IR is validated before ptr boundary

- **WHEN** PTO 输入通过 VPTO backend lowering 和 post-lowering cleanup 进入发射前阶段
- **THEN** backend MUST 在 `PTOVPTOPtrBoundary` 之前运行 authoring-stage VPTO legality verifier
- **AND** 该 verifier MUST 看见 backend mainline 收敛后的 VPTO IR，而不是 `PTOToVPTO` 之后的早期过渡形态

#### Scenario: direct VPTO input is not allowed to bypass authoring-stage validation

- **WHEN** 输入文件本身已经是 VPTO IR，导致 backend lowering 主线被跳过
- **THEN** `--pto-backend=vpto` 路径仍 MUST 运行 authoring-stage VPTO legality verifier
- **AND** MUST NOT 仅因为输入已经是 VPTO IR 就直接进入 emission boundary

### Requirement: VPTO vector and predicate structure MUST stay inside a single `llvm.loop.aivector_scope`

所有 VPTO 向量 / 谓词 / align 相关 op MUST 位于某个带 `llvm.loop.aivector_scope` 的 `scf.for` 作用域内。  
同时，带 `llvm.loop.aivector_scope` 的 carrier loop MUST NOT 嵌套另一个同样带该 attr 的 carrier loop。

#### Scenario: vector or predicate VPTO op outside vec scope is rejected

- **WHEN** authoring-form VPTO IR 中出现消费或产生 `!pto.vreg`、`!pto.mask<...>`、`!pto.align` 的 VPTO op，且该 op 不在任何 `llvm.loop.aivector_scope` loop 内
- **THEN** authoring-stage verifier MUST 拒绝该 IR
- **AND** 诊断 MUST 明确指出违规 op 位于 vec scope 外

#### Scenario: nested vec scopes are rejected

- **WHEN** 某个带 `llvm.loop.aivector_scope` 的 `scf.for` 作用域内再次出现带同样 attr 的 `scf.for`
- **THEN** authoring-stage verifier MUST 拒绝该 IR
- **AND** 诊断 MUST 明确指出存在 nested vec scope

#### Scenario: shared scalar and control-flow surface is still allowed outside vec scope

- **WHEN** `arith`、`scf`、pointer-building、copy programming 或 sync programming 相关 op 本身不产生也不消费 `!pto.vreg`、`!pto.mask<...>`、`!pto.align`
- **THEN** authoring-stage verifier MUST NOT 仅因为这些 op 位于 vec scope 外就拒绝 IR
- **AND** vec scope 约束 MUST 只针对完整版 `vpto-spec.md` 中定义的 VPTO vector / predicate / align surface 生效

### Requirement: VPTO masks MUST encode granularity in the type and match consuming vector families

VPTO mask 类型 MUST 显式写成 `!pto.mask<b8>`、`!pto.mask<b16>` 或 `!pto.mask<b32>`。  
消费向量的 VPTO op MUST 要求 mask type 与向量元素家族一致，而不是依赖 opaque mask 或运行时约定。  
同时，名字显式编码 granularity 的 predicate family MUST 与类型参数一致。

#### Scenario: typed mask with mismatched vector element width is rejected

- **WHEN** `!pto.mask<b16>` 被用于消费 `f32/i32` 向量的 VPTO op
- **THEN** authoring-stage verifier MUST 拒绝该 IR
- **AND** 诊断 MUST 指出实际 mask type 与 consumer vector element type 不匹配

#### Scenario: predicate family suffix must match typed mask result

- **WHEN** `pset_b32`、`pge_b32`、`plt_b32` 等显式 `b32` family 产出非 `!pto.mask<b32>` 的结果类型
- **THEN** verifier MUST 拒绝该 IR
- **AND** 诊断 MUST 指出 family granularity 与 mask type 不一致

#### Scenario: compare and carry families must preserve one mask granularity

- **WHEN** `vcmp` / `vcmps` 的 seed mask、result mask 与输入向量元素家族不一致，或 `vaddc` / `vsubc` / `vaddcs` / `vsubcs` 的 `mask` / `carry` / `carry_in` 之间出现 granularity 不一致
- **THEN** authoring-stage verifier MUST 拒绝该 IR
- **AND** 诊断 MUST 指出是 compare family 还是 carry family 违反了 granularity contract

#### Scenario: mask-only and predicate-movement families must preserve declared granularity

- **WHEN** `pnot` / `psel` / `ppack` / `punpack` 这类 mask-only family 的输入输出 `G` 不一致，或 `pdintlv_b8` / `pintlv_b16` 这类 predicate movement family 的类型参数与 family 自身编码的 granularity 不一致
- **THEN** authoring-stage verifier MUST 拒绝该 IR
- **AND** 诊断 MUST 指出是 mask-only family 还是 predicate movement family 违反了 granularity contract

#### Scenario: function or block arguments carrying masks must stay typed

- **WHEN** 函数参数、block 参数或 region 参数承载 VPTO mask
- **THEN** 这些值 MUST 显式写成 `!pto.mask<b8>`、`!pto.mask<b16>` 或 `!pto.mask<b32>`
- **AND** authoring-stage verifier MUST NOT 以“opaque mask 来源”为理由跳过 granularity 校验

### Requirement: VPTO address-form legality MUST follow the `copy` / `buffer-like` / `ptr-only` split defined by `vpto-spec.md`

VPTO address-form legality MUST follow the `copy` / `buffer-like` / `ptr-only` split defined by `vpto-spec.md`.

完整版 `vpto-spec.md` 已将 VPTO 地址形态分成三类：  
`copy family` 继续要求 typed `!pto.ptr`；  
stateless / predicate `buffer-like family` 在 authoring form 可接受 `memref<...>` 或 `!pto.ptr<...>`；  
stateful `%base/%base_out` family 保持 pointer-only。  
OpenSpec 的 legality contract MUST 沿用这一定义，而不是在实现中另起一套口径。

#### Scenario: authoring-form buffer-like family is not rejected merely for using memref

- **WHEN** stateless / predicate `buffer-like family` 在 authoring form 使用 `memref<...>` 作为地址操作数，且其他局部签名约束合法
- **THEN** authoring-stage verifier MUST NOT 仅因为其不是 ptr-form 就拒绝该 IR
- **AND** 这类 memref-first surface MUST 被视为合法的 VPTO authoring contract 一部分

#### Scenario: ptr-only family remains pointer-only before and after emission

- **WHEN** `vlds_post` / `vsts_post` / `vldas` / `vldus` / `pstu` / `vstu` / `vstus` / `vstur` 等 pointer-only family 出现 memref address operand
- **THEN** IR MUST 被拒绝
- **AND** 该拒绝可以继续由现有单-op verifier 负责，而不是要求模块级 legality verifier 重新实现一遍

### Requirement: VPTO emission boundary MUST revalidate ptr-form IR before any final emit

在 `PTOVPTOPtrBoundary` 将 IR canonicalize 为 ptr-form 之后，发射路径 MUST 再执行一次 emission-stage VPTO legality verifier。  
该阶段除了保留 authoring-stage 的结构规则外，还 MUST 验证最终 emission contract：不得残留 memref function boundary、不得残留 memref-form emission op、不得残留仍参与 emission 的 dead scaffold。

#### Scenario: final ptr-form IR is validated before `--emit-vpto`

- **WHEN** CLI 路径在 `PTOVPTOPtrBoundary` 之后准备输出最终 VPTO IR
- **THEN** emission-stage verifier MUST 在 `--emit-vpto` 输出之前运行
- **AND** 输出的最终 VPTO IR MUST 已通过 ptr-form emission legality 验证

#### Scenario: final ptr-form IR is validated before text and LLVM emission

- **WHEN** CLI 或 direct API 路径准备进行 text emission 或 LLVM emission
- **THEN** emission-stage verifier MUST 在真正发射之前运行
- **AND** text emission、LLVM text emission 和 LLVM bitcode emission MUST 共享同一套 ptr-form emission legality contract

#### Scenario: residual memref boundary or memref-form emission op is rejected

- **WHEN** `PTOVPTOPtrBoundary` 之后仍残留 memref function argument、memref function result、受支持 buffer-like VPTO op 的 memref-form 地址操作数，或仍参与 emission 的 `pto.bind_tile` / 平凡 `pto.castptr` / `memref.subview` / `memref.reinterpret_cast` / `memref.memory_space_cast`
- **THEN** emission-stage verifier MUST 拒绝该 IR
- **AND** 诊断 MUST 明确指出残留的 boundary、op 或 scaffold 类型

