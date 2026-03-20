# A5 OpLib V1 作者接口与对齐说明

- 状态：Current
- 适用范围：PTOAS A5 OpLib V1
- 目标读者：`oplib/level3` 模板维护者、`PTOLowerToOpLibCalls` / tile fusion 维护者、A5 范围内新 family 的实现者

本文是 A5 OpLib V1 的当前作者接口说明。对外可导入契约仍然是 checked-in concrete `.mlir` 模板，但作者的主维护入口已经切换为 Family DSL、snippet 合同和 A5 manifest 对齐。

## 1. 范围边界

A5 OpLib V1 的首批范围固定为 [`docs/PTO_IR_manual.md`](../PTO_IR_manual.md) 第 4.5~4.9 节：

1. 4.5 `Vector Arithmetic Operations`
2. 4.6 `Reduction Operations`
3. 4.7 `Broadcast Operations`
4. 4.8 `Compare & Select Operations`
5. 4.9 `Bitwise Operations`

当前 V1 in-scope op 集固定如下：

1. 4.5
   `tadd`, `tsub`, `tmul`, `tdiv`, `tmax`, `tmin`, `trem`,
   `tpartadd`, `tpartmax`, `tpartmin`, `tprelu`,
   `tadds`, `tsubs`, `tmuls`, `tdivs`, `tmaxs`, `tmins`, `trems`,
   `taddc`, `tsubc`, `taddsc`, `tsubsc`,
   `tabs`, `tneg`, `texp`, `tlog`, `tsqrt`, `trsqrt`, `trecip`, `trelu`, `tlrelu`
2. 4.6
   `trowsum`, `trowmax`, `trowmin`, `tcolsum`, `tcolmax`, `tcolmin`
3. 4.7
   `trowexpand`, `tcolexpand`, `trowexpandmul`, `trowexpanddiv`, `trowexpandsub`, `texpands`
4. 4.8
   `tcmp`, `tcmps`, `tsel`, `tsels`
5. 4.9
   `tand`, `tor`, `txor`, `tshl`, `tshr`, `tnot`

以下内容明确不属于 V1：

1. 4.5~4.9 之外的 manual section
2. `tload` / `tstore` / `tmov` / `ttrans` / `tsync`
3. `tmatmul` / `tgemv` 等 matrix compute
4. 新的 Level-1 / Level-2 公共作者 DSL
5. 在 `pto-isa` 仓内承载 MLIR compiler 入口

## 2. 当前真值层与文件布局

A5 OpLib V1 的主维护源分成四类文件：

1. Family DSL：
   [`oplib/level3/families/a5_oplib_v1_family_dsl.json`](../../oplib/level3/families/a5_oplib_v1_family_dsl.json)
2. snippet 合同：
   [`oplib/level3/families/a5_oplib_v1_snippet_contracts.json`](../../oplib/level3/families/a5_oplib_v1_snippet_contracts.json)
3. family snippet：
   [`oplib/level3/families/snippets/`](../../oplib/level3/families/snippets)
4. A5 manifest snapshot：
   [`oplib/level3/families/a5_oplib_v1_manifest.yaml`](../../oplib/level3/families/a5_oplib_v1_manifest.yaml)

生成产物和消费入口如下：

1. concrete 模板输出：
   [`oplib/level3/`](../../oplib/level3)
2. 生成入口：
   [`oplib/level3/generate_level3_templates.py`](../../oplib/level3/generate_level3_templates.py)
3. lowering/importer 消费侧：
   `ptoas` 默认自动查找 installed/repo `oplib/level3`，也支持
   `ptoas --op-lib-dir=<concrete-template-dir>` 显式覆盖

工程约束固定如下：

1. 作者不直接把 concrete `.mlir` 当作主维护源。
2. lowering 不直接解释 Family DSL 源，而是继续导入 checked-in concrete `.mlir`。
3. manifest 是 4.5~4.9 范围、A5 状态、dtype 约束和语义来源路径的真值层。

## 3. 作者接口

当前作者接口是“Family DSL + snippet”，而不是“手写完整 concrete 函数体”。

### 3.1 Family DSL 负责什么

Family DSL 负责声明：

1. family / `kind`
2. 参数角色和 ABI 形状
3. 支持的 op 列表
4. dtype 轴、variant 轴
5. `cmpMode` / `scalarPos` / `requiredVariantId` / `isBinary` 等 matcher 轴
6. family 级 metadata、cost、priority
7. 与 manifest 的映射关系

### 3.2 snippet 负责什么

snippet 只负责核心 Mixed-Body MLIR 计算逻辑，不负责重复样板：

1. 不手写 concrete `func.func` 命名
2. 不手写全量 `pto.oplib.*` metadata
3. 不手写固定 `tile_to_memref`
4. 不手写统一 tail mask / `vector.maskedload` / `vector.maskedstore` 骨架
5. 不因 `dtype`、`op`、condition、variant 变化而复制整份函数体

### 3.3 generator 负责什么

generator 固定负责：

1. 校验 Family DSL、snippet 合同和 catalog 投影
2. 展开参数列表和 concrete `func.func`
3. 写入 `pto.oplib.kind`、`pto.oplib.entry_role`、`pto.oplib.op`、`pto.oplib.variant_id`
4. 写入 `pto.oplib.match.*`
5. 生成统一的 `tile_to_memref`、`pto.simd.vec_scope`、tail mask 和 64-lane SIMD 骨架
6. 输出 importer-active concrete `.mlir`

## 4. Manifest 对齐规则

manifest 仍然只允许两种外部状态：

1. `implemented`
2. `deferred`

但作者和 lowering 侧必须进一步区分语义来源，而不能再把 `implemented` 简化成“有模板即可”。

### 4.1 语义证据来源

A5 OpLib V1 当前承认三类语义证据，优先级如下：

1. `include/pto/npu/a5/*.hpp` 中的 A5 原生语义，例如 `*_IMPL`、`OP_NAME(...)`
2. `include/pto/common/pto_instr.hpp` 中已经稳定的公共 API 重写语义
3. `tests/npu/a5/src/st/testcase/` 中可确认的 A5 ST 用例

manifest 校验与作者决策都必须基于这三类证据，而不能只看 A5 原生头文件。

### 4.2 `implemented`

若某个 op 在 manifest 中标记为 `implemented`，则必须同时满足：

1. 该 op 属于固定 4.5~4.9 in-scope 集合
2. 其语义来源属于以下两类之一：
   `native_a5_impl` 或已批准的 `public_api_rewrite`
3. manifest 中声明的每个 `dtype_support` 都必须能在 concrete 模板中找到对应覆盖，而不是仅“至少有一个模板”
4. lowering 必须能基于 manifest + template registry 为这些 dtype 完成实例选择、实例创建和 call rewrite
5. 必须存在对应回归，防止“manifest 已实现但 dtype / variant 级模板或 lowering 静默掉队”

当前唯一已批准的 `public_api_rewrite` 是 `trecip`：

1. `pto/common/pto_instr.hpp` 将其稳定重写为 `TDIVS(dst, 1, src)` 语义
2. manifest 应保持 `implemented`
3. OpLib template / matcher / lowering 允许按该等价语义进入 implemented 集

### 4.3 `deferred`

若某个 op 在 manifest 中标记为 `deferred`，则必须同时满足：

1. 仍然保留在 manifest 中
2. `deferred_reason` 非空
3. lowering 对该 op 给出确定性失败，并在诊断中保留 `deferred_reason`
4. 不能通过“暂时没有模板”来静默缺位

当前 `taddc`、`tsubc`、`taddsc`、`tsubsc` 仍属于这种情况：

1. 它们可以在 EmitC 路径通过 decomposition 保持行为
2. 但由于缺少被当前 change 接受的 A5 implemented 语义来源，不能因此自动进入 OpLib implemented 集
3. 若未来要把 decomposition 提升为 OpLib implemented 语义，必须单独补文档、模板和回归

## 5. Lowering 与 matcher 契约

A5 OpLib V1 不改变 importer 合约，只改变作者维护方式。

`PTOLowerToOpLibCalls` 继续消费 concrete 模板，并继续使用以下 matcher key：

1. `kind`
2. `op`
3. `dtype`
4. `variant_id`
5. `cmpMode`
6. `scalarPos`
7. `requiredVariantId`
8. `isBinary`

对实现者而言，这意味着：

1. 修改 Family DSL 或 snippet 时，不能私自引入新的匹配协议。
2. compare/select family 仍要保持旧 `cmpMode` / scalar-mode 语义。
3. reduction / tile-scalar 等 family 仍要保持 `requiredVariantId` / `isBinary` 的选择语义。

## 6. 推荐作者工作流

新增或修改某个 family 时，建议按以下顺序操作：

1. 先确认该 op 是否在 manifest 的 4.5~4.9 in-scope 集内。
2. 若 manifest 仍是 `deferred`，先补 manifest 对齐，而不是直接私加模板。
3. 修改 Family DSL、snippet 合同或 family snippet。
4. 运行 generator / Family DSL 校验。
5. 生成并检查 concrete `.mlir`。
6. 同步确认 manifest 的 `a5_status`、`dtype_support`、`deferred_reason` 与当前语义证据一致。
7. 补对应 positive / negative / 对齐测试。
8. 再检查 lowering、tile fusion、EmitC 的相关回归。

常用命令：

```bash
python3 oplib/level3/family_dsl.py --check-snippet-contracts
python3 oplib/level3/family_dsl.py --check-catalog
python3 oplib/level3/generate_level3_templates.py --check
python3 test/oplib/check_a5_manifest_semantics.py \
  --manifest=oplib/level3/families/a5_oplib_v1_manifest.yaml
python3 test/oplib/check_implemented_op_alignment.py \
  --manifest=test/oplib/resources/implemented_op_alignment_subset.json \
  --template-dir=oplib/level3 \
  --test-dir=test/oplib
llvm-lit -sv \
  test/oplib/oplib_manifest_semantic_alignment.mlir \
  test/oplib/oplib_compare_select_integer_dtypes.mlir \
  test/oplib/oplib_reduction_broadcast_dtype_coverage.mlir \
  test/oplib/oplib_arith_bitwise_dtype_coverage.mlir \
  test/oplib/oplib_manifest_deferred.mlir \
  test/oplib/oplib_trecip_public_api_rewrite.mlir \
  test/oplib/oplib_key_family_dtype_lowering_smoke.mlir \
  test/oplib/oplib_v1_negative_regressions.mlir
```

说明：

1. `check_a5_manifest_semantics.py` 负责校验 manifest 与同级 `pto-isa` 语义证据是否一致，包括 `native_a5_impl`、`public_api_rewrite` 和 `missing_accepted_semantics` 三类。
2. `check_implemented_op_alignment.py` 现阶段按 dtype 粒度工作，但默认使用仓内维护的 subset manifest 做门禁，避免把尚未补齐 lowering use case 的 implemented 全量快照直接绑死。
3. 全量 implemented 集的 dtype / lowering 覆盖应继续通过 family 专项 lit 和 smoke 回归逐步补齐。

## 7. 文档与实现边界

当前文档拆分建议如下：

1. 本文是 A5 OpLib V1 的当前作者接口和边界说明。
2. [`docs/tile_fusion/oplib_ir_spec.md`](./oplib_ir_spec.md) 继续记录多 family importer / metadata 契约。
3. [`docs/tile_fusion/oplib_simd_template_min_guide.md`](./oplib_simd_template_min_guide.md) 保留 mixed IR / SIMD 模板体约束的最小指南，但不再代表完整作者入口。

若文档、manifest、generator 或 lowering 之间出现不一致，以当前 checked-in manifest、Family DSL 和 concrete 回归为准，并优先回到 OpenSpec change 进行修正。
