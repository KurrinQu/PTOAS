## ADDED Requirements

### Requirement: A5 OpLib V1 authoring must use a declaration-first family template model

A5 OpLib V1 模板体系 MUST 以声明式 Family DSL 作为主维护源。模板作者 SHALL 维护 family 级 schema 和 Mixed-Body MLIR snippet，系统 SHALL 生成 importer-active concrete `.mlir` 模板；系统 MUST NOT 要求作者继续直接维护仅在 `dtype`、`opname`、condition 或 variant 上不同的重复 concrete 函数体。

#### Scenario: Author maintains family spec and snippet instead of concrete boilerplate

- **WHEN** 模板作者为某个 in-scope family 新增或修改实现
- **THEN** 作者输入 MUST 由 family spec 和 snippet 组成，generator MUST 负责生成 concrete `func.func`、统一 metadata、循环骨架、tail mask 和 SIMD load/store 样板

#### Scenario: Concrete templates remain importer-active outputs

- **WHEN** generator 根据 family spec 和 snippet 产出模板文件
- **THEN** 输出 MUST 仍然是 checked-in concrete `.mlir` 模板，并继续作为 `--op-lib-dir` 的 importer 输入

### Requirement: A5 OpLib V1 template scope must be limited to PTO IR manual sections 4.5-4.9

A5 OpLib V1 的首批模板覆盖范围 MUST 只包括 `PTO_IR_manual.md` 第 4.5~4.9 节的 op 集，不得默认扩张到其他 manual section。

#### Scenario: In-scope operator set is explicit

- **WHEN** V1 定义首批模板覆盖范围
- **THEN** in-scope op 集 MUST 显式包含以下算子：
  - 4.5: `tadd`, `tsub`, `tmul`, `tdiv`, `tmax`, `tmin`, `trem`, `tpartadd`, `tpartmax`, `tpartmin`, `tprelu`, `tadds`, `tsubs`, `tmuls`, `tdivs`, `tmaxs`, `tmins`, `trems`, `taddc`, `tsubc`, `taddsc`, `tsubsc`, `tabs`, `tneg`, `texp`, `tlog`, `tsqrt`, `trsqrt`, `trecip`, `trelu`, `tlrelu`
  - 4.6: `trowsum`, `trowmax`, `trowmin`, `tcolsum`, `tcolmax`, `tcolmin`
  - 4.7: `trowexpand`, `tcolexpand`, `trowexpandmul`, `trowexpanddiv`, `trowexpandsub`, `texpands`
  - 4.8: `tcmp`, `tcmps`, `tsel`, `tsels`
  - 4.9: `tand`, `tor`, `txor`, `tshl`, `tshr`, `tnot`

#### Scenario: Out-of-scope operators are not pulled into V1 by template infra

- **WHEN** 某个 op 不属于 4.5~4.9 节
- **THEN** A5 OpLib V1 模板 authoring / generator 契约 MUST NOT 以“顺便支持”的方式把该 op 纳入 V1 范围

### Requirement: A5 OpLib V1 legality and coverage must follow a PTO-ISA-aligned A5 manifest

A5 OpLib V1 模板覆盖矩阵 MUST 以 `pto-isa` A5 自动对齐得到的 manifest snapshot 为真值。manifest SHALL 显式标记 `implemented` 或 `deferred` 状态，并记录 dtype / layout / tmp / mask / variant 等关键约束。

#### Scenario: Implemented operator generates concrete templates

- **WHEN** 某个 4.5~4.9 op 在 A5 manifest 中被标记为 `implemented`
- **THEN** 系统 MUST 为该 op 生成 concrete 模板候选，并要求后续存在对应 lowering 回归

#### Scenario: Deferred operator remains explicit

- **WHEN** 某个 4.5~4.9 op 在 A5 manifest 中被标记为 `deferred`
- **THEN** 系统 MAY 暂不为其生成模板候选，但 MUST 保留显式状态和缺失原因，且 MUST NOT 静默遗漏该 op

### Requirement: A5 OpLib V1 generated bodies must stay within the existing SIMD/vector authoring envelope

A5 OpLib V1 生成出的模板体 MUST 继续基于现有 `pto.simd.*` 与 `vector/arith/memref/scf/math` 允许集合构造，不得默认引入新的 public authoring 层。

#### Scenario: Existing simd envelope is reused by default

- **WHEN** generator 生成某个 in-scope family 的 concrete 模板
- **THEN** 模板体 MUST 优先复用现有 `pto.simd.tile_to_memref`、`pto.simd.vec_scope`、`pto.simd.predicate/load/store/load_pu/store_pu` 以及既有 `vector/arith/memref/scf/math` 组合

#### Scenario: New simd primitive requires explicit extension

- **WHEN** 某个 4.5~4.9 op 无法在现有 authoring 集合中表达
- **THEN** 实现 MAY 扩展最小的新 primitive，但该扩展 MUST 伴随 verifier、lowering 和 EmitC 规则的同步定义，而不能只在模板层私自使用
