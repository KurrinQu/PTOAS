# OpLib Lowering 与 Tile Fusion 设计说明（A5 Level-3）

- 状态：Current
- 适用范围：`ptoas --pto-arch=a5` 的当前源码实现
- 目标读者：`ptoas` pipeline 维护者、`PTOLowerToOpLibCalls` / tile fusion 维护者、`oplib/level3` 模板维护者

## 1. 文档范围

本文说明 A5 Level-3 下当前仓库中 `OpLib lowering + tile fusion` 的实现结构、模板组织方式、桥接 IR 设计、pass 顺序与边界条件。

本文只描述当前实现，不讨论未来 proposal。本文使用以下文件作为真值层：

1. [`tools/ptoas/ptoas.cpp`](../../tools/ptoas/ptoas.cpp)
2. [`include/PTO/IR/PTOInterfaces.td`](../../include/PTO/IR/PTOInterfaces.td)
3. [`include/PTO/IR/PTOOpLibMatch.h`](../../include/PTO/IR/PTOOpLibMatch.h)
4. [`include/PTO/IR/PTOOps.td`](../../include/PTO/IR/PTOOps.td)
5. [`include/PTO/Transforms/Passes.td`](../../include/PTO/Transforms/Passes.td)
6. [`lib/PTO/IR/PTO.cpp`](../../lib/PTO/IR/PTO.cpp)
7. [`lib/PTO/Transforms/PTOCreateFusionGroups.cpp`](../../lib/PTO/Transforms/PTOCreateFusionGroups.cpp)
8. [`lib/PTO/Transforms/PTOOutlineFusionGroups.cpp`](../../lib/PTO/Transforms/PTOOutlineFusionGroups.cpp)
9. [`lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp`](../../lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp)
10. [`lib/PTO/Transforms/PTOLowLevelLoopFusion.cpp`](../../lib/PTO/Transforms/PTOLowLevelLoopFusion.cpp)
11. [`lib/PTO/Transforms/PTOViewToMemref.cpp`](../../lib/PTO/Transforms/PTOViewToMemref.cpp)
12. [`lib/PTO/Transforms/PTOToEmitC.cpp`](../../lib/PTO/Transforms/PTOToEmitC.cpp)
13. [`oplib/level3/skeletons/README.md`](../../oplib/level3/skeletons/README.md)
14. [`docs/tile_fusion/a5_oplib_v1_authoring.md`](./a5_oplib_v1_authoring.md)
15. [`docs/tile_fusion/oplib_ir_spec.md`](./oplib_ir_spec.md)
16. [`oplib/level3/families/a5_oplib_v1_family_dsl.json`](../../oplib/level3/families/a5_oplib_v1_family_dsl.json)
17. [`oplib/level3/families/a5_oplib_v1_manifest.yaml`](../../oplib/level3/families/a5_oplib_v1_manifest.yaml)

## 2. 术语与层次

本文使用以下术语区分不同层次的范围：

| 术语                        | 含义                                                                         | 当前状态                                                                     |
| --------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Manifest / Family Scope     | A5 OpLib V1 在 4.5~4.9 范围内承认的 op family、dtype 与状态                  | 范围较宽，覆盖 arithmetic / reduction / broadcast / compare-select / bitwise |
| Match Descriptor Scope      | 已实现 `OpLibOpInterface::getOpLibMatchDescriptor()` 的 PTO op 范围          | 大于当前 active lowering 范围                                                |
| Active OpLib Lowering Scope | `PTOLowerToOpLibCalls.cpp` 中 `shouldLowerViaOpLib()` 当前实际改写的 op 范围 | 当前仅 12 个 arithmetic / tile-scalar op                                     |
| Tile Fusion Grouping Scope  | `PTOCreateFusionGroups` 当前允许分组的 op 范围                               | 与 active lowering 的 12 个 op 一致                                          |

当前实现中，manifest 范围、descriptor 范围、active lowering 范围、tile fusion grouping 范围并不相同。文档后续所有结论都以此分层为前提。

## 3. 总体架构

当前 A5 Level-3 方案由两层机制组成：

1. `OpLib lowering`
   将单个 PTO op 映射到已导入模板中的一个具体实例，再改写为 `func.call @__pto_oplib_inst_*`。
2. `tile fusion`
   在高层 PTO arithmetic 链上做连续分组与 outline，在 helper 内完成 grouped lowering，再在内联后的低层 loop 上做结构化融合。

总体路径如下：

```text
PTO IR op
  │
  ├─ OpLibOpInterface::getOpLibMatchDescriptor()
  │     生成 family-aware 匹配请求
  │
  ├─ A5 manifest + imported concrete templates
  │     进行 family / dtype / variant / attr matching
  │
  ├─ 路径 A：single-op OpLib lowering
  │     PTOInstantiateAndLowerToLibCall
  │       -> func.call @__pto_oplib_inst_*
  │     PTOInlineLibCall
  │       -> inline 后得到 mixed IR / vector IR
  │
  └─ 路径 B：tile fusion
        PTOCreateFusionGroups
          -> 写入 pto.fusion.group_id / pto.fusion.order
        PTOOutlineFusionGroups
          -> 生成 @__pto_fused_group_*
        PTOInstantiateAndLowerToLibCall
          -> helper 内逐 op lower 成 OpLib call
        PTOInlineLibCall
          -> helper 内出现相邻 loop / vec_scope
        PTOLowLevelLoopFusion
          -> 对 helper 内低层 loop 做结构化融合
```

该架构的关键事实如下：

1. `OpLib lowering` 是 A5 base pipeline 的一部分。
2. `tile fusion` 由 `--enable-op-fusion` 额外打开。
3. `PTOLowLevelLoopFusion` 只处理 synthesized fused helper，不处理普通函数中的 single-op lowering 结果。

## 4. OpLib Family Skeleton 设计

## 4.1 设计目标

`oplib/level3` 的 family skeleton 设计用于解决以下问题：

1. 同一 family 内往往共享相同的外围结构，包括 tile 到 memref 的桥接、二维循环、64-lane mask 构造、`pto.simd.vec_scope` 包装与 store 回写。
2. 同一 family 的差异通常集中在少数轴上，例如 `op`、`dtype`、`variant_id`、`cmpMode`、`operandOrder`、`isBinary`。
3. 直接维护大量 importer-active concrete `.mlir` 模板会导致重复代码过多，难以保持 matcher metadata、外部 ABI 和循环骨架的一致性。
4. lowering / importer 侧已经稳定依赖 concrete `func.func` 与 `pto.oplib.*` 元数据，不适合直接改为消费 DSL 源或 snippet 源。

因此当前实现采用“单一维护源 + concrete 生成”的分层结构。

## 4.2 当前维护层次

根据 [`oplib/level3/skeletons/README.md`](../../oplib/level3/skeletons/README.md) 与作者文档，当前维护层次如下：

| 层次              | 位置                                                                                                           | 职责                                                               |
| ----------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Family DSL        | [`oplib/level3/families/a5_oplib_v1_family_dsl.json`](../../oplib/level3/families/a5_oplib_v1_family_dsl.json) | 定义 family、参数角色、dtype 轴、variant 轴、matcher key、输出文件 |
| Snippet Contract  | `oplib/level3/families/a5_oplib_v1_snippet_contracts.json`                                                     | 定义 snippet 可见 SSA 名称与插槽合同                               |
| Snippet           | [`oplib/level3/families/snippets/`](../../oplib/level3/families/snippets)                                      | 只维护核心计算表达式                                               |
| Skeleton Source   | [`oplib/level3/skeletons/`](../../oplib/level3/skeletons)                                                      | 维护统一的外围 mixed IR 骨架                                       |
| Concrete Template | [`oplib/level3/`](../../oplib/level3)                                                                          | importer-active 输入，供 `ptoas --op-lib-dir=...` 直接导入         |

当前 lowering 只消费 concrete template，不直接消费 Family DSL、snippet 或 skeleton source。

## 4.3 Skeleton 解决的问题

以 tile-scalar family 为例，骨架负责统一处理：

1. 外部 ABI 形态
2. `pto.oplib.*` 元数据框架
3. `pto.simd.tile_to_memref` 桥接
4. `memref.dim` 与二维循环
5. 64-lane mask、passive vector 与 `vector.maskedload/store`
6. `pto.simd.vec_scope` 包装

对应的 skeleton 片段如下：

```mlir
func.func private @@@FUNC_NAME@@(
@@ARG_DECLS@@
    ) attributes {
      pto.oplib.kind = "@@KIND@@",
      pto.oplib.op = "@@OP_NAME@@",
      pto.oplib.variant_id = "@@VARIANT_ID@@",
      pto.oplib.match.dtype = "@@MATCH_DTYPE@@",
      pto.oplib.match.scalar_pos = @@SCALAR_POS@@ : i64,
@@MATCH_ATTRS@@
    } {
  %m0 = pto.simd.tile_to_memref %src0 : @@INPUT_TILE_TYPE@@ to @@INPUT_MEMREF_TYPE@@
  %md = pto.simd.tile_to_memref %dst : @@RESULT_TILE_TYPE@@ to @@RESULT_MEMREF_TYPE@@
  %rows = memref.dim %m0, %c0 : @@INPUT_MEMREF_TYPE@@
  %cols = memref.dim %m0, %c1 : @@INPUT_MEMREF_TYPE@@
  pto.simd.vec_scope {
    %passive = arith.constant @@PASSIVE_VECTOR@@ : @@VECTOR_TYPE@@
    %scalarVec = vector.splat %scalar : @@VECTOR_TYPE@@
@@EXTRA_SETUP@@
    scf.for %r = %c0 to %rows step %c1 {
      scf.for %cidx = %c0 to %cols step %c64 {
        %mask = vector.create_mask %active : @@MASK_VECTOR_TYPE@@
        %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive ...
@@COMPUTE@@
        vector.maskedstore %md[%r, %cidx], %mask, %result ...
      }
    }
  }
  return
}
```

该骨架中，真正按 op 变化的部分被收敛到 `@@COMPUTE@@` 插槽。

## 4.4 Snippet 负责的内容

tile-scalar family 的 elementwise snippet 仅负责核心计算表达式：

```mlir
%result = @@CORE_OP@@ @@SNIPPET_LHS@@, @@SNIPPET_RHS@@@@EXEC_MODE_ATTR@@ : @@VECTOR_TYPE@@
```

该分层将“外围循环 / mask / bridge / metadata”和“核心算子表达式”解耦。对于 `tadds`、`tsubs`、`tmuls`、`tdivs` 这类 op，生成器只需要在 `@@CORE_OP@@`、`variant_id`、附加 matcher 轴上展开，而不需要复制整份函数体。

## 4.5 Family DSL 负责的内容

以 `float_tile_scalar` family 为例，当前 DSL 负责声明：

1. `kind = "l3_float_tile_scalar_template"`
2. 参数角色为 `(tile, scalar, tile)`
3. `dtype_axis`
4. `variant_axis`
5. `matcher_keys`
6. 各个 op 对应的 `core_op` 与 snippet

当前 DSL 片段如下：

```json
{
  "family": "float_tile_scalar",
  "pattern": "tile_scalar",
  "kind": "l3_float_tile_scalar_template",
  "snippet_contract": "tile_scalar_result",
  "parameter_roles": [
    { "name": "src", "kind": "tile", "io": "input" },
    { "name": "scalar", "kind": "scalar", "io": "input" },
    { "name": "dst", "kind": "tile", "io": "output" }
  ],
  "matcher_keys": [
    { "key": "kind", "attr": "pto.oplib.kind" },
    { "key": "op", "attr": "pto.oplib.op" },
    { "key": "dtype", "attr": "pto.oplib.match.dtype" },
    { "key": "variant_id", "attr": "pto.oplib.variant_id" },
    { "key": "scalarPos", "attr": "pto.oplib.match.scalar_pos" },
    { "key": "requiredVariantId", "location": "request_only" }
  ]
}
```

该设计将 family 的结构性差异前移到生成阶段，而将 lowering 侧保持为对 concrete 模板与 matcher key 的消费。

## 4.6 Concrete Template 的角色

生成后的 concrete template 是当前 lowering 的直接输入。以 `tadds/f32` 为例：

```mlir
func.func private @__pto_oplib_variant_tadds_f32(
    %src0: !pto.tile_buf<...>,
    %scalar: f32,
    %dst: !pto.tile_buf<...>
    ) attributes {
      pto.oplib.kind = "l3_float_tile_scalar_template",
      pto.oplib.entry_role = "variant",
      pto.oplib.op = "tadds",
      pto.oplib.variant_id = "tile_scalar",
      pto.oplib.match.dtype = "f32",
      pto.oplib.match.scalar_pos = 1 : i64
    } {
  %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<...> to memref<?x?xf32, ...>
  %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<...> to memref<?x?xf32, ...>
  pto.simd.vec_scope {
    %scalarVec = vector.splat %scalar : vector<64xf32>
    %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive
           {pto.simd.vld_dist = "NORM"} : ...
    %result = arith.addf %lhs, %scalarVec
              {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>
    vector.maskedstore %md[%r, %cidx], %mask, %result
      {pto.simd.vst_dist = "DIST_NORM"} : ...
  }
  return
}
```

该 concrete template 同时承载：

1. importer 所需的 `pto.oplib.*` 元数据
2. A5 vector lowering 所需的 mixed IR 体
3. EmitC 所需的 `pto.simd.*` 属性

## 5. 对接 CCE / EmitC 的桥接 IR 设计

## 5.1 设计目标

当前实现并不直接把 OpLib 模板写成 CCE C++ 字符串，也不在模板阶段提前降到目标平台内建调用。当前采用的是“mixed MLIR + `pto.simd` bridge op”的设计：

1. 外部 ABI 维持 tile-oriented 视图。
2. 模板体内部使用 `vector`、`arith`、`scf`、`memref` 与少量 `pto.simd` bridge op。
3. 与 A5/CCE 相关的 backend 语义通过 `pto.simd` 标记和属性保留到 `PTOToEmitC`。

## 5.2 关键桥接点

当前实现中，以下 bridge op 和属性直接承担 CCE/EmitC 对接职责：

| IR 元素                   | 作用                                        | 后端语义                                          |
| ------------------------- | ------------------------------------------- | ------------------------------------------------- |
| `pto.simd.vec_scope`      | 标记一段必须在 `__VEC_SCOPE__` 下执行的区域 | `PTOToEmitC` 中直接变成 `__VEC_SCOPE__ { ... }`   |
| `pto.simd.tile_to_memref` | 保留从 tile-like 值到 memref 视图的桥接标记 | EmitC 阶段据此物化 `tile.data()` 或等价指针       |
| `pto.simd.vld_dist`       | 记录 load 分发方式                          | EmitC 中转成 `vlds(..., NORM)` 等 load 语义       |
| `pto.simd.vst_dist`       | 记录 store 分发方式                         | EmitC 中转成 `vsts(..., DIST_NORM)` 等 store 语义 |
| `pto.simd.exec_mode`      | 记录向量算术执行模式                        | EmitC 中转成 `MODE_ZEROING` 等模式参数            |

## 5.3 `pto.simd.tile_to_memref` 为何保留到 EmitC

`PTOViewToMemref.cpp` 中明确保留 `pto.simd.tile_to_memref` 作为 backend marker。注释中的设计意图是：

1. 该桥接不在 `PTOViewToMemref` 阶段完全消除。
2. EmitC 阶段再物化 `tile.data()`，以便锚定在 tile 绑定完成之后。
3. 对 op-fusion 路径中的 helper / instance remap，仍保留统一的桥接语义。

在 `PTOLowerToOpLibCalls.cpp` 的实例化逻辑中，即使模板体已经进入 memref-world，`simd.tile_to_memref` 仍会作为 backend marker 被重新建立，而不是直接消除。

## 5.4 `pto.simd.vec_scope` 到 `__VEC_SCOPE__`

`include/PTO/IR/PTOOps.td` 与 `lib/PTO/Transforms/PTOToEmitC.cpp` 当前定义了非常直接的映射：

```mlir
pto.simd.vec_scope {
  ...
}
```

在 EmitC lowering 之后变成：

```c++
__VEC_SCOPE__ {
  ...
}
```

这一设计使模板作者仍然在 MLIR 层表达循环和向量语义，同时保留 A5 vector scope 的后端边界。

## 5.5 从 mixed IR 到 CCE 风格代码的映射

对当前 active family，内联后的 helper IR 一般具有以下形态：

```mlir
pto.simd.vec_scope {
  %lhs = vector.maskedload ...
  %rhs = vector.maskedload ...
  %result = arith.addf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>
  vector.maskedstore ...
}
```

对应 EmitC 输出则体现为 CCE 风格的 API / intrinsic：

```c++
__VEC_SCOPE__ {
  vlds(v28, v1, v27, NORM);
  vlds(v30, v2, v29, NORM);
  vadd(v31, v28, v30, v26, MODE_ZEROING);
  vsts(v31, v7, v32, ..., v26);
}
```

因此，当前设计的桥接层不是“独立的 CCE dialect”，而是由：

1. `pto.simd` 标记
2. 标准 `vector/arith/scf/memref`
3. `PTOToEmitC` 中的专门 conversion pattern

共同构成的 backend bridge。

## 6. Pipeline 与 Pass 顺序

## 6.1 开关语义

`tools/ptoas/ptoas.cpp` 中当前开关语义如下：

1. `enableA5OplibPipeline = (effectiveArch == PTOTargetArch::A5)`
2. A5 目标下默认解析 installed/repo `oplib/level3`，也支持显式传入 `--op-lib-dir`
3. `--enable-op-fusion` 只控制 fusion 相关 pass，不控制 base OpLib lowering
4. `--pto-arch!=a5` 时，`--enable-op-fusion` 被忽略

因此：

| 条件                                        | 当前行为                                                             |
| ------------------------------------------- | -------------------------------------------------------------------- |
| `--pto-arch=a5` 且未传 `--enable-op-fusion` | 仍会导入 OpLib 模板并执行 single-op OpLib lowering                   |
| `--pto-arch=a5 --enable-op-fusion`          | 在 base OpLib lowering 之外，附加 group / outline / low-level fusion |
| `--pto-arch=a3 --enable-op-fusion`          | 不进入该链路                                                         |

## 6.2 当前与本文相关的 pass 顺序

| 阶段    | 顺序 | Pass / 步骤                       | 条件                      | 作用                                                             |
| ------- | ---- | --------------------------------- | ------------------------- | ---------------------------------------------------------------- |
| Stage 0 | 1    | `importPTOOpLibTemplates`         | A5                        | 导入 concrete template                                           |
| Stage 1 | 1    | `LoweringSyncToPipe`              | 总是                      | 处理同步管线语义                                                 |
| Stage 1 | 2    | `PTOValidateSimdIR`               | A5                        | 校验模板体与 SIMD 桥接 IR                                        |
| Stage 1 | 3    | `PTOViewToMemref`                 | 总是                      | 进入 memref-world，同时保留 `simd.tile_to_memref` backend marker |
| Stage 2 | 1    | `InferPTOLayout`                  | 默认开启                  | 推断布局                                                         |
| Stage 2 | 2    | `PTOViewToMemref`                 | 总是                      | 进一步收敛 memref-world 表示                                     |
| Stage 2 | 3    | `PlanMemory`                      | 非 Level-3                | 本地内存规划                                                     |
| Stage 2 | 4    | `PTOInsertSync`                   | 条件开启                  | 插入 sync；对 grouping 构成断链点                                |
| Stage 2 | 5    | `PTOCreateFusionGroups`           | A5 + `--enable-op-fusion` | 标注连续 arithmetic chain                                        |
| Stage 2 | 6    | `PTOOutlineFusionGroups`          | A5 + `--enable-op-fusion` | 生成 `@__pto_fused_group_*`                                      |
| Stage 2 | 7    | `PTOInstantiateAndLowerToLibCall` | A5                        | single-op / grouped lowering                                     |
| Stage 2 | 8    | `PTOInlineLibCall`                | A5                        | 内联实例函数                                                     |
| Stage 2 | 9    | `PTOLowLevelLoopFusion`           | A5 + `--enable-op-fusion` | 对 fused helper 的低层 loop 进行融合                             |
| Stage 2 | 10   | `Canonicalizer + CSE`             | A5                        | 在 low-level fusion 之后做清理                                   |
| Stage 2 | 11   | `Canonicalizer + CSE`             | 总是                      | 通用清理                                                         |

当前实现刻意将 A5 内部那一轮 `Canonicalizer + CSE` 放在 `PTOLowLevelLoopFusion` 之后，以避免 single-trip loop 被过早折叠，破坏 low-level fusion 所依赖的规则结构。

## 7. 匹配契约

## 7.1 `OpLibOpInterface`

`include/PTO/IR/PTOInterfaces.td` 定义：

```tablegen
def OpLibOpInterface : OpInterface<"OpLibOpInterface"> {
  let methods = [
    InterfaceMethod<
      "Build OP-Lib match descriptor for this op.",
      "::mlir::FailureOr<::mlir::pto::OpLibMatchDescriptor>",
      "getOpLibMatchDescriptor",
      (ins)
    >
  ];
}
```

该接口要求每个参与 OpLib lowering 的 PTO op 自行导出 family-aware descriptor。当前 lowering 不依赖 pass 内部的大量 family-specific `dyn_cast` 分支来恢复语义。

## 7.2 `OpLibMatchDescriptor`

当前 descriptor 字段包括：

| 字段                              | 含义                             |
| --------------------------------- | -------------------------------- |
| `kind`                            | family key                       |
| `opName`                          | PTO mnemonic                     |
| `operands`                        | 匹配与改写使用的实参序列         |
| `operandRoles`                    | 每个参数是 `Tile` 还是 `Scalar`  |
| `maskContract`                    | compare/select 的 mask 合同标记  |
| `operandOrder`                    | 顺序敏感 family 的额外轴         |
| `fullTilePos` / `rowBroadcastPos` | broadcast-row-binary 的角色位置  |
| `scalarPos`                       | scalar 参数位置                  |
| `cmpMode`                         | compare 模式                     |
| `isBinary`                        | reduction / broadcast 特定语义位 |
| `requiredVariantId`               | 必须命中的模板变体               |

## 7.3 family-specific 特殊字段

当前源码中已有多类 family 的 descriptor 实现。重要的 family-specific 约束如下：

| family / op                                         | 额外字段                            | 当前含义                                                         |
| --------------------------------------------------- | ----------------------------------- | ---------------------------------------------------------------- |
| `tdivs`                                             | `operandOrder`, `requiredVariantId` | 区分 `tile_scalar` 与 `scalar_tile`，并对 `f16/i16` 追加变体后缀 |
| `tcmp`, `tcmps`                                     | `maskContract`, `cmpMode`           | compare family 要求 canonical byte-mask 语义                     |
| `tsel`, `tsels`                                     | 无额外字段或使用上游 compare 结果   | 依赖 select family 的固定签名与 mask / mode 语义                 |
| `tcolsum`                                           | `isBinary`, `requiredVariantId`     | 区分 `binary` 与 `linear` 变体                                   |
| `trowexpandmul` / `trowexpanddiv` / `trowexpandsub` | `fullTilePos`, `rowBroadcastPos`    | 标识 full tile 与 row source 的角色位置                          |

## 7.4 descriptor 范围与 active lowering 范围

当前 `lib/PTO/IR/PTO.cpp` 中已实现 descriptor 的 family 范围明显大于 active lowering 范围，当前实现状态如下：

| 类别                       | 代表 op                                                                                                            | 状态                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| binary tile-tile           | `tadd`, `tsub`, `tmul`, `tdiv`, `tmax`, `tmin`, `trem`                                                             | descriptor 已实现；active lowering 当前只放开前 6 个 |
| tile-scalar                | `tadds`, `tsubs`, `tmuls`, `tdivs`, `tmaxs`, `tmins`, `trems`                                                      | descriptor 已实现；active lowering 当前只放开前 6 个 |
| unary / partial arithmetic | `tabs`, `tneg`, `trelu`, `texp`, `tlog`, `tsqrt`, `trsqrt`, `trecip`, `tprelu`, `tpartadd`, `tpartmax`, `tpartmin` | descriptor 已实现；当前不走 active lowering          |
| compare / select           | `tcmp`, `tcmps`, `tsel`, `tsels`                                                                                   | descriptor 已实现；当前不走 active lowering          |
| reduction                  | `trowsum`, `trowmax`, `trowmin`, `tcolmax`, `tcolmin`, `tcolsum`                                                   | descriptor 已实现；当前不走 active lowering          |
| broadcast                  | `trowexpand`, `tcolexpand`, `trowexpandmul`, `trowexpanddiv`, `trowexpandsub`, `texpands`                          | descriptor 已实现；当前不走 active lowering          |
| bitwise                    | `tand`, `tor`, `txor`, `tshl`, `tshr`, `tnot`                                                                      | descriptor 已实现；当前不走 active lowering          |

## 8. memref-world 与模板 ABI

当前实现存在两个并存的 ABI 视图：

| 层次                          | 当前表示                                         |
| ----------------------------- | ------------------------------------------------ |
| 模板作者 / 导入层             | 以 `!pto.tile_buf` 为主导的 template ABI         |
| OpLib matching / rewriting 层 | 允许已 lower 为 `memref` 的 tile-like 值参与匹配 |

`PTOViewToMemref` 之后，PTO arithmetic op 经常已经位于 memref-world。`test/tile_fusion/memref_to_tilebuf_alloc.mlir` 明确校验了这一点。与此同时，`simd.tile_to_memref` 仍被保留为 backend marker，并在实例化阶段根据实参类型重新建立。

这一设计保证了：

1. 模板维护接口仍然稳定。
2. base pipeline 可以在 memref-world 完成 instance selection 与 rewrite。
3. EmitC 阶段仍能恢复 tile-oriented backend 语义。

## 9. 分阶段 IR 转化

本节使用当前仓内 `test/oplib/softmax_chain.pto` 的 pass dump 与 codegen 输出作为示例。片段已省略无关常量、布局属性与部分参数，仅保留与本设计相关的关键结构。

## 9.1 memref-world PTO IR

在 `PTOViewToMemref` 之后，算子已经在 memref-world 上运行：

```mlir
%1 = pto.bind_tile %0, %c32, %c32 ... :
     memref<32x32xf32, ...> -> memref<32x32xf32, ..., #pto.address_space<vec>>
%5 = pto.bind_tile %4, %c32, %c32 ... :
     memref<32x32xf32, ...> -> memref<32x32xf32, ..., #pto.address_space<vec>>

pto.tmuls ins(%1, %cst : memref<32x32xf32, ...>, f32)
          outs(%1 : memref<32x32xf32, ...>)
pto.tmaxs ins(%1, %cst_0 : memref<32x32xf32, ...>, f32)
          outs(%1 : memref<32x32xf32, ...>)
pto.tmul ins(%1, %1 : memref<32x32xf32, ...>, memref<32x32xf32, ...>)
         outs(%5 : memref<32x32xf32, ...>)
```

该阶段的重要事实是：

1. 参与 grouping / lowering 的 PTO op 已经使用 memref 作为 tile-like carrier。
2. 这并不阻止后续匹配模板，因为 active lowering 允许 lowered memref ABI。

## 9.2 `PTOCreateFusionGroups`

group 创建之后，连续 arithmetic chain 被写入 `pto.fusion.group_id` 与 `pto.fusion.order`：

```mlir
pto.tmuls ins(%1, %cst : memref<32x32xf32, ...>, f32)
          outs(%1 : memref<32x32xf32, ...>)
          {pto.fusion.group_id = 0 : i64, pto.fusion.order = 0 : i64}
pto.tmaxs ins(%1, %cst_0 : memref<32x32xf32, ...>, f32)
          outs(%1 : memref<32x32xf32, ...>)
          {pto.fusion.group_id = 0 : i64, pto.fusion.order = 1 : i64}
pto.tmins ins(%1, %cst_1 : memref<32x32xf32, ...>, f32)
          outs(%1 : memref<32x32xf32, ...>)
          {pto.fusion.group_id = 0 : i64, pto.fusion.order = 2 : i64}
pto.tmul ins(%1, %1 : memref<32x32xf32, ...>, memref<32x32xf32, ...>)
         outs(%5 : memref<32x32xf32, ...>)
         {pto.fusion.group_id = 0 : i64, pto.fusion.order = 3 : i64}
```

该 pass 不做语句重排；它只在同一 block 内识别严格连续的 producer-consumer 链。

## 9.3 `PTOOutlineFusionGroups`

outline 之后，原 block 中的连续链被替换成对 helper 的调用：

```mlir
call @__pto_fused_group_0_0(%cst, %cst_0, %cst_1, %1, %5, %7, %9)
  : (f32, f32, f32, memref<32x32xf32, ...>, memref<32x32xf32, ...>,
     memref<32x32xf32, ...>, memref<32x32xf32, ...>) -> ()

call @__pto_fused_group_1_1(%1, %5, %7, %9, %cst_2, %cst_3, %3)
  : (memref<32x32xf32, ...>, memref<32x32xf32, ...>, memref<32x32xf32, ...>,
     memref<32x32xf32, ...>, f32, f32, memref<32x32xf32, ...>) -> ()
```

helper 的参数由两部分组成：

1. `externalInputs`
2. `producedInOrder`

因此 helper 更接近“对原地写 buffer 链的封装函数”，而不是返回 SSA value 的纯函数。

## 9.4 `PTOInstantiateAndLowerToLibCall`

在 helper 内部，PTO op 被逐个改写为实例函数调用：

```mlir
func.func private @__pto_fused_group_0_0(
    %arg0: f32, %arg1: f32, %arg2: f32,
    %arg3: memref<32x32xf32, ...>, %arg4: memref<32x32xf32, ...>,
    %arg5: memref<32x32xf32, ...>, %arg6: memref<32x32xf32, ...>) {
  call @__pto_oplib_inst_l3_float_tile_scalar_template_tmuls_tile_scalar(
      %arg3, %arg0, %arg3) : (...)
  call @__pto_oplib_inst_l3_float_tile_scalar_template_tmaxs_tile_scalar(
      %arg3, %arg1, %arg3) : (...)
  call @__pto_oplib_inst_l3_float_tile_scalar_template_tmins_tile_scalar(
      %arg3, %arg2, %arg3) : (...)
  call @__pto_oplib_inst_l3_float_binary_elementwise_template_tmul_tile(
      %arg3, %arg3, %arg4) : (...)
  return
}
```

该阶段保留了 helper 结构，但高层 PTO op 已被替换为 instance call。

## 9.5 `PTOInlineLibCall`

内联之后，helper 中出现 mixed IR / vector IR：

```mlir
func.func private @__pto_fused_group_1_1(..., %arg4: f32, %arg5: f32, %arg6: memref<32x32xf32, ...>) {
  %0 = pto.simd.tile_to_memref %arg0 : memref<32x32xf32, ...> to memref<32x32xf32, ...>
  %1 = pto.simd.tile_to_memref %arg1 : memref<32x32xf32, ...> to memref<32x32xf32, ...>
  %2 = pto.simd.tile_to_memref %arg6 : memref<32x32xf32, ...> to memref<32x32xf32, ...>
  pto.simd.vec_scope {
    %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
    scf.for %r = %c0 to %dim step %c1 {
      scf.for %cidx = %c0 to %dim_0 step %c64 {
        %mask = vector.create_mask %active : vector<64xi1>
        %lhs = vector.maskedload %0[%r, %cidx], %mask, %cst
               {pto.simd.vld_dist = "NORM"} : ...
        %rhs = vector.maskedload %1[%r, %cidx], %mask, %cst
               {pto.simd.vld_dist = "NORM"} : ...
        %sum = arith.addf %lhs, %rhs
               {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>
        vector.maskedstore %2[%r, %cidx], %mask, %sum
          {pto.simd.vst_dist = "DIST_NORM"} : ...
      }
    }
  }
  ...
}
```

该阶段为 `PTOLowLevelLoopFusion` 提供了规则化的 loop / `vec_scope` 结构。

## 9.6 EmitC / CCE 风格输出

在 `PTOToEmitC` 之后，`__pto_fused_group_1_1` 的输出片段如下：

```c++
[aicore] inline __attribute__((always_inline)) void __pto_fused_group_1_1(
    __ubuf__ float* v1, __ubuf__ float* v2, __ubuf__ float* v3,
    __ubuf__ float* v4, float v5, float v6, __ubuf__ float* v7) {
  __VEC_SCOPE__ {
    ...
    vlds(v28, v1, v27, NORM);
    vlds(v30, v2, v29, NORM);
    vadd(v31, v28, v30, v26, MODE_ZEROING);
    ...
    vadd(v40, v37, v19, v39, MODE_ZEROING);
    vdiv(v43, v40, v22, v42, MODE_ZEROING);
    vsts(v43, v7, v44, ..., v26);
  }
}
```

当前 codegen 已经清楚体现出以下对应关系：

1. `pto.simd.vec_scope` -> `__VEC_SCOPE__`
2. `vector.maskedload` + `pto.simd.vld_dist` -> `vlds`
3. `arith.addf` / `arith.divf` + `pto.simd.exec_mode` -> `vadd` / `vdiv`
4. `vector.maskedstore` + `pto.simd.vst_dist` -> `vsts`

## 10. 当前 active lowering 范围

当前 `PTOLowerToOpLibCalls.cpp` 中 `shouldLowerViaOpLib()` 将 active lowering 范围限制为 12 个 op：

| 类别                   | op                                                   |
| ---------------------- | ---------------------------------------------------- |
| tile-tile arithmetic   | `tadd`, `tsub`, `tmul`, `tdiv`, `tmax`, `tmin`       |
| tile-scalar arithmetic | `tadds`, `tsubs`, `tmuls`, `tdivs`, `tmaxs`, `tmins` |

这一范围同时构成：

1. 当前 single-op OpLib lowering 的 active 范围
2. 当前 tile fusion grouping 的 active 范围

对这 12 个 active op，`PTOInstantiateAndLowerToLibCall` 在改写结束后还会检查是否存在“应被 lower 但未被 lower”的残留 op。若存在残留，pass 会直接报错，而不是静默回退。

## 11. 当前 manifest 与 descriptor 的更宽范围

尽管 active lowering 仅有上述 12 个 op，当前 manifest 与 descriptor 已覆盖更宽的 family 范围：

1. unary / partial arithmetic
2. compare / select
3. reduction
4. broadcast
5. bitwise
6. ternary family

其中，`taddc`、`tsubc`、`taddsc`、`tsubsc` 在当前 manifest 语义中仍属于 deferred 范围，不属于 active implemented 集。

## 12. Tile Fusion 设计

## 12.1 `PTOCreateFusionGroups`

当前 `PTOCreateFusionGroups` 的分组条件如下：

1. op 必须实现 `OpLibOpInterface`
2. op 必须实现 `PTO_DpsInitOpInterface`
3. `getOpLibMatchDescriptor()` 必须成功
4. `desc.opName` 必须位于 12-op allowlist 中
5. op 只能有一个 DPS init
6. descriptor 的最后一个 operand 必须是 `dst`
7. 当前 op 的 tile 输入必须包含前一个 op 的 `dst`
8. 任何 intervening op 都会打断 chain

该 pass 只在 block 内做线性扫描，不进行跨 block 或 DAG 级重组。

## 12.2 `PTOOutlineFusionGroups`

当前 outline 逻辑将同一 `group_id` 且 `order` 连续的 op 收集为一个 group，并生成：

```text
func.func private @__pto_fused_group_<group_id>_<counter>
```

helper 接口按 `externalInputs` + `producedInOrder` 拼接参数列表。helper 自身带有 `pto.fusion.group_id` 属性。

## 12.3 grouped lowering

grouped lowering 与 single-op lowering 共享同一套 matcher / selector / instance builder。区别仅在于：

1. single-op 路径遍历普通函数中未分组的 op
2. grouped 路径在 helper 中按 `group_id + order` 收集后逐个改写

当前 grouped lowering 不支持“部分成功”。如果某个组内 op 无法完成 plan / rewrite，整个 group 降低失败。

## 13. Low-Level Loop Fusion 设计

## 13.1 处理对象

`PTOLowLevelLoopFusion` 当前只处理函数名以 `__pto_fused_group_` 开头的 helper：

```text
func.func @__pto_fused_group_*
```

它不处理：

1. 普通函数中的 single-op lowering 结果
2. 未经过 outline 的代码

## 13.2 处理结构

当前 pass 的描述与实现均表明，它处理的是：

1. 相邻的 canonical `pto.simd.vec_scope` stage
2. helper 内联后出现的规则 loop nest
3. 与中间 stage result 相关的 store-to-load forwarding

pass 内显式分析 `vector.maskedstore` / `vector.maskedload` 模式，并在条件满足时消除中间存取。

## 13.3 触发条件

low-level loop fusion 是结构化模式匹配，不是通用 loop fusion 框架。当前通常要求：

1. 相邻阶段的 loop nest 形状兼容
2. mask 兼容
3. stage 结构满足 pass 预期
4. producer / consumer 关系落在支持的 forwarding 模式内

shape、mask 或嵌套结构不匹配时，pass 保守地保持原状。

## 14. 当前边界

当前实现具有以下边界：

1. A5 manifest 范围不等于 active lowering 范围。
2. descriptor 已实现不等于 pipeline 已启用。
3. tile fusion grouping 只覆盖 12 个 arithmetic / tile-scalar op。
4. grouping 只处理严格连续的 block 内 chain，不做 DAG 级融合。
5. low-level loop fusion 只处理 synthesized fused helper。
6. backend bridge 依赖 `pto.simd` 标记与 EmitC conversion pattern，不是独立的 CCE dialect。

## 15. 阅读与维护顺序

建议按照以下顺序理解和维护当前方案：

1. 先阅读 [`docs/tile_fusion/a5_oplib_v1_authoring.md`](./a5_oplib_v1_authoring.md) 与 [`docs/tile_fusion/oplib_ir_spec.md`](./oplib_ir_spec.md)，明确作者接口与 family 边界。
2. 再阅读 [`oplib/level3/skeletons/README.md`](../../oplib/level3/skeletons/README.md)、[`oplib/level3/skeletons/`](../../oplib/level3/skeletons) 与 [`oplib/level3/families/a5_oplib_v1_family_dsl.json`](../../oplib/level3/families/a5_oplib_v1_family_dsl.json)，明确 skeleton / snippet / concrete 的分层。
3. 然后阅读 [`lib/PTO/IR/PTO.cpp`](../../lib/PTO/IR/PTO.cpp) 中各个 op 的 `getOpLibMatchDescriptor()` 实现，明确 family-specific matcher 字段。
4. 再阅读 [`lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp`](../../lib/PTO/Transforms/PTOLowerToOpLibCalls.cpp)，明确 active gate、template import、instance selection、instance clone 与 rewrite。
5. 最后阅读 [`lib/PTO/Transforms/PTOCreateFusionGroups.cpp`](../../lib/PTO/Transforms/PTOCreateFusionGroups.cpp)、[`lib/PTO/Transforms/PTOOutlineFusionGroups.cpp`](../../lib/PTO/Transforms/PTOOutlineFusionGroups.cpp)、[`lib/PTO/Transforms/PTOLowLevelLoopFusion.cpp`](../../lib/PTO/Transforms/PTOLowLevelLoopFusion.cpp)，明确 fusion 链条。

建议配合以下测试阅读：

1. [`test/tile_fusion/memref_to_tilebuf_alloc.mlir`](../../test/tile_fusion/memref_to_tilebuf_alloc.mlir)
2. [`test/tile_fusion/create_fusion_groups.mlir`](../../test/tile_fusion/create_fusion_groups.mlir)
3. [`test/tile_fusion/materialize_fusion_groups.mlir`](../../test/tile_fusion/materialize_fusion_groups.mlir)
4. [`test/oplib/oplib_grouped_lowering_chain_coverage.mlir`](../../test/oplib/oplib_grouped_lowering_chain_coverage.mlir)
5. [`test/tile_fusion/low_level_loop_fusion.mlir`](../../test/tile_fusion/low_level_loop_fusion.mlir)
6. [`test/tile_fusion/emitc_fused_helper_codegen.mlir`](../../test/tile_fusion/emitc_fused_helper_codegen.mlir)

## 16. 小结

当前 A5 Level-3 的 `OpLib lowering + tile fusion` 方案可概括为：

1. A5 下始终启用 base OpLib pipeline；`--enable-op-fusion` 仅附加 fusion pass。
2. `oplib/level3` 采用 Family DSL、snippet、skeleton、concrete template 的分层组织，以减少模板重复并保持 importer 接口稳定。
3. 模板体采用 mixed IR + `pto.simd` bridge 的设计，通过 `PTOToEmitC` 对接 `__VEC_SCOPE__` 与 CCE 风格向量 API。
4. 当前 active lowering 与 grouping 仅覆盖 12 个 arithmetic / tile-scalar op。
5. low-level loop fusion 发生在 outlined helper 内联之后的低层 loop 结构上，而不是发生在高层 PTO op 本身上。
