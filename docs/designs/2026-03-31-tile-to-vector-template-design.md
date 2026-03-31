# Tile→Vector 模板实例化与向量库方案设计


## 第一章 背景与问题

### 1.1 IR 的层级关系

PTOAS 编译栈分为三层，Tile IR 面向用户，Vector IR 面向硬件，LLVM IR 由后端接管：

```
用户代码 / 上层框架
        │
        ▼
┌──────────────────────┐
│  PTO Tile IR         │  pto.tadd, pto.tmul, pto.tload ...
│  操作对象: tile_buf   │  形状/dtype 由 tile_buf 类型携带
└──────────┬───────────┘
           │  Tile→Vector Lowering
           ▼
┌──────────────────────┐
│  Vector IR (vPTO)    │  pto.vadd, pto.vlds, pto.vsts ...
│  操作对象: vreg/ptr  │  循环遍历 tile，每次处理一个 vreg
└──────────┬───────────┘
           │  Vector→LLVM Lowering
           ▼
┌──────────────────────┐
│  LLVM IR             │  由 LLVM 编译器接管
└──────────────────────┘
```

Tile IR 中一条 op 表达完整的 tile 语义，Vector IR 则必须显式展开为循环、按 vreg 宽度分块、处理尾部 mask。

### 1.2 用 `pto.tadd` 看两层之间的差距

**Tile IR 层**——一条指令，语义完整：

```mlir
pto.tadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>)
```

**Vector IR 层**——同一个 `pto.tadd` 的向量实现（`dtype=f32, rows=16, cols=64`）：

```mlir
func.func @TADD(
    %a: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %b: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %c: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    attributes { pto.tile_function = "pto.tadd" } {
  %vecA = pto.tile_buf_addr %a : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
      v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %vecB = pto.tile_buf_addr %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
      v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %vecC = pto.tile_buf_addr %c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
      v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index

  pto.vecscope {
    scf.for %arg0 = %c0 to %c16 step %c1 {           // 遍历 rows
      scf.for %arg1 = %c0 to %c64 step %c64 {         // 遍历 cols，步长=vector_width
        %va = pto.vlds %vecA[%arg0, %arg1] : memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vb = pto.vlds %vecB[%arg0, %arg1] : memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vc = pto.vadd %va, %vb : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>
        pto.vsts %vc, %vecC[%arg0, %arg1] : !pto.vreg<64xf32>, memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
      }
    }
  }
  return
}
```

从这段向量实现可以看到：`dtype=f32` 决定了 vreg 类型 `vreg<64xf32>`（64 = 256B / sizeof(f32)）；`rows=16, cols=64` 决定了 memref 形状和循环边界；`blayout` 决定了 stride 模式。这些值贯穿整个函数体。

### 1.3 问题的本质

`pto.tadd` 只是一个例子，所有 Tile 指令（`pto.tsub`、`pto.tmul`、`pto.tneg` ...）都面临相同问题：

- 实际使用中 `tile_buf` 的参数组合千变万化——dtype 有 f16/f32/bf16 等，rows/cols 可以是任意正整数。
- 为每种 `(op, dtype, rows, cols, layout)` 组合手写向量实现不可行。
- 需要一种方案：按 `dtype` 分模板函数，shape 和布局用占位符表示，由 lowering 在编译期用具体 `tile_buf` 类型特化。

## 第二章 方案与模板库

### 2.1 总体思路

将每个 Tile 指令的 Vector 实现写成模板函数，组成向量库：

1. **按 `(tile op, dtype)` 组织**：`dtype` 决定 vreg 元素类型和向量宽度，必须固化。种类有限（f16/f32/bf16/i8/i16/i32），每种一个模板。
2. **shape 用 intrinsic 占位**：`rows/cols/v_row/v_col` 不写死，通过 intrinsic op 从 `tile_buf` 类型中提取。
3. **编译期特化**：Lowering 遇到 Tile op 时，选择模板、克隆、用实际 `tile_buf` 类型特化、inline 到调用点。

### 2.2 模板库接口约定

| 项目 | 约定 |
|------|------|
| 函数命名 | `@{OP}_{dtype}`，如 `@TADD_f32`、`@TMUL_f16` |
| 函数属性 | `pto.tile_function = "pto.tadd"` 标记对应的 Tile 指令 |
| 参数顺序 | 与 Tile op ODS 定义一致，`ins` 在前，`outs` 在后 |
| 参数类型 | `!pto.tile_buf<..., rows=?, cols=?, v_row=?, v_col=?, ...>` 通用模板签名 |
| 读写属性 | `pto.access = "read" / "write" |

**Tile 属性 Intrinsic**——在模板中作为占位符使用，特化后 fold 为常量：

| Intrinsic | 语义 | 返回类型 |
|-----------|------|----------|
| `pto.tile_rows %tb` | 提取 rows | `index` |
| `pto.tile_cols %tb` | 提取 cols | `index` |
| `pto.tile_valid_rows %tb` | 提取 v_row | `index` |
| `pto.tile_valid_cols %tb` | 提取 v_col | `index` |
| `pto.tile_blayout %tb` | 提取 blayout 编码 | `i32` |
| `pto.tile_slayout %tb` | 提取 slayout 编码 | `i32` |
| `pto.tile_fractal %tb` | 提取 fractal size | `i32` |
| `pto.tile_pad %tb` | 提取 pad 策略编码 | `i32` |

**`pto.tile_buf_addr`**——从 `tile_buf` 提取 memref，输出类型在特化阶段由 tile_buf 类型自动推导：

```
特化前: pto.tile_buf_addr %src0 : tile_buf<..., rows=?, cols=?> -> memref<?x?xf32, strided<[?, 1]>, #pto.address_space<vec>>
特化后: pto.tile_buf_addr %src0 : tile_buf<..., rows=16, cols=64, blayout=row_major, loc=vec>
        -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
```

推导规则：`rows/cols` → memref shape；`blayout=row_major` → `strided<[cols, 1]>`，`col_major` → `strided<[1, rows]>`；`loc` → `#pto.address_space<loc>`。

### 2.3 用 `TADD_f32` 说明模板怎么写

```mlir
func.func @TADD_f32(
    %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=?, cols=?, v_row=?, v_col=?,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>
           {pto.access = "read"},
    %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=?, cols=?, v_row=?, v_col=?,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>
           {pto.access = "read"},
    %dst:  !pto.tile_buf<loc=vec, dtype=f32, rows=?, cols=?, v_row=?, v_col=?,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>
           {pto.access = "write"})
    attributes { pto.tile_function = "pto.tadd" } {

  // 1. 从 tile_buf 提取 memref（shape 和 stride 为占位，特化后推导）
  %mSrc0 = pto.tile_buf_addr %src0 : ... -> memref<?x?xf32, strided<[?, 1]>, #pto.address_space<vec>>
  %mSrc1 = pto.tile_buf_addr %src1 : ... -> memref<?x?xf32, strided<[?, 1]>, #pto.address_space<vec>>
  %mDst  = pto.tile_buf_addr %dst  : ... -> memref<?x?xf32, strided<[?, 1]>, #pto.address_space<vec>>

  // 2. 通过 intrinsic 提取有效形状（占位，特化后 fold 为常量）
  %v_rows = pto.tile_valid_rows %src0 : ... -> index
  %v_cols = pto.tile_valid_cols %src0 : ... -> index
  %v_cols_i32 = arith.index_cast %v_cols : index to i32  // plt_b32 需要 i32

  // 3. dtype=f32 → vector_width=64（256B / 4B），这是固化在模板中的
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  // 4. 向量循环体：按行遍历，按 vreg 宽度分块，带尾部 mask
  pto.vecscope {
    scf.for %i = %c0 to %v_rows step %c1 {
      scf.for %j = %c0 to %v_cols step %c64 iter_args(%remain = %v_cols_i32) -> (i32) {
        %mask, %next = pto.plt_b32 %remain : i32 -> !pto.mask, i32

        %va = pto.vlds %mSrc0[%i, %j]
            : memref<?x?xf32, strided<[?, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vb = pto.vlds %mSrc1[%i, %j]
            : memref<?x?xf32, strided<[?, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vc = pto.vadd %va, %vb, %mask
            : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
        pto.vsts %vc, %mDst[%i, %j], %mask
            : !pto.vreg<64xf32>, memref<?x?xf32, strided<[?, 1]>, #pto.address_space<vec>>, !pto.mask

        scf.yield %next : i32
      }
    }
  }
  return
}
```

这个模板体现的写法要点：

1. `f32` 对应的 vreg 类型 `!pto.vreg<64xf32>` 和向量宽度 `%c64` 是固化的——这是按 dtype 分模板的原因。
2. `rows/cols` 不写死，通过 `tile_valid_rows/tile_valid_cols` 提取，特化后成为常量。
3. `tile_buf_addr` 把 `tile_buf` 转成 memref，特化后 shape/stride 自动推导。
4. 尾块处理通过 `pto.plt_b32` + mask 显式表达。
5. 整个函数本质上就是一段"可被具体 tile_buf 类型特化的 Vector IR"。

### 2.4 特化与 Fold

模板特化是编译期行为。当某个调用点的实际类型为 `tile_buf<..., rows=16, cols=64, v_row=16, v_col=64, ...>` 时：

1. 模板参数类型被替换为具体 `tile_buf` 类型。
2. `tile_buf_addr` 结果类型推导为 `memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>`。
3. 静态字段的 intrinsic fold 为 `arith.constant`：

```cpp
OpFoldResult TileRowsOp::fold(FoldAdaptor adaptor) {
  auto tbType = getOperand().getType().cast<TileBufType>();
  int64_t rows = tbType.getShape()[0];
  if (rows == ShapedType::kDynamic)
    return {};  // 动态字段，无法折叠
  return IntegerAttr::get(IndexType::get(getContext()), rows);
}
```

4. 动态字段（如 `v_row=?`）的 intrinsic 保留，inline 到调用点后由 canonicalize 在上下文中解析：

```mlir
// inline 后（调用点处）：
%rows = arith.constant 16 : index       // rows 静态，已折叠
%cols = arith.constant 64 : index       // cols 静态，已折叠
%v_row = pto.tile_valid_rows %a : ...   // v_row 动态，保留 intrinsic
```

### 2.5 模板库的约束

1. **Intrinsic 只允许出现在顶层 `pto.tile_function` 函数中。** 模板内部的辅助函数不直接使用 intrinsic，而是通过参数接收已提取的值。这保证特化逻辑只处理顶层函数，无需递归分析。
2. **模板库内置到编译器中**（编译期链接的 MLIR module）。Lowering pass 初始化时加载，通过 `pto.tile_function` 属性值 + dtype 匹配模板。

## 第三章 编译器方案：Tile→Vector Lowering

### 3.1 整体 Pass Pipeline

```
ConvertToPTOOp → tile ops (tile_buf)
       ↓
PTOViewToMemref            ← tile_buf → memref
       ↓
PlanMemory (memref)        ← 保持不变
       ↓
InsertSync (memref)        ← 保持不变
       ↓
  ┌─── 路径选择 ────────────────────┐
  │                                │
  ▼                                ▼
MemrefToTileBuf (new pass)     PTOToEmitC / A5VM (memref)
  ↓
VF Fusion Analysis (tile_buf)
  ↓
Tile→Vector Lowering (tile_buf)
  ↓
Vector Fusion / Cleanup
  ↓
Vector→LLVM Lowering
```

关键设计决策：

- **Tile→Vector lowering 必须在 `tile_buf` 级别工作**，因为模板中的 intrinsic 依赖 `tile_buf` 类型信息。
- **PlanMemory / InsertSync 继续在 memref 上工作**（短期方案），在 InsertSync 之后通过 `MemrefToTileBuf` pass 将 memref 转回 tile_buf。
- **VF 融合分析在 Tile IR 层完成**，基于 `ins/outs` 数据流和 tile_buf 类型兼容性判断。
- **模板 inline 后不保留 `func.call` 边界**，使后续向量循环融合可以跨原 Tile op 边界执行。

> 长期方向是将 PlanMemory / InsertSync 迁移为基于 tile_buf 工作，两条路径共享，但当前改造量较大，作为后续独立演进目标。

### 3.2 Lowering 核心步骤

以 `pto.tadd ins(%a, %b) outs(%c)` 为例，`%a/%b/%c` 类型为 `tile_buf<f32, 16, 64, ...>`：

```
Step 1: 识别 Tile op，读取 dtype → f32
Step 2: 查找 pto.tile_function="pto.tadd" 且 dtype=f32 的模板 → @TADD_f32
Step 3: 克隆模板，用实际 tile_buf 类型替换签名中的动态类型 (rows=16, cols=64, ...)
Step 4: 推导 tile_buf_addr 的结果 memref 类型
Step 5: Fold intrinsic
         - 静态字段：pto.tile_rows → arith.constant 16
         - 动态字段：pto.tile_valid_rows → 保留
Step 6: 将特化后的函数体 inline 到原 Tile op 位置
Step 7: Cleanup / canonicalize
```

输出——原 `pto.tadd` 被替换为向量循环体：

```mlir
pto.vecscope {
  scf.for %i = ... {
    scf.for %j = ... {
      %va = pto.vlds ...
      %vb = pto.vlds ...
      %vc = pto.vadd %va, %vb, %mask ...
      pto.vsts %vc, ...
    }
  }
}
```

### 3.3 与融合的关系

VF 融合分析在 Tile IR 层完成：

```mlir
pto.tadd ins(%a, %b) outs(%tmp)    // tmp = a + b
pto.tmul ins(%tmp, %d) outs(%c)    // c = tmp * d
// %tmp 是 tadd 的输出、tmul 的输入，类型兼容 → 标记为可融合
```

Tile→Vector lowering 将两个 op 分别 inline 为向量循环体后，Vector Fusion pass 根据融合标记合并相邻循环，消除 `%tmp` 的 UB 读写，减少 `vlds/vsts` 和 VF 发射次数。

## 第四章 前置工作

### 4.1 `TileBufType` parser 支持 `rows=?`, `cols=?`

当前 parser 只对 `v_row`/`v_col` 支持 `?`（`parseOptionalQuestion` → `kDynamic`），`rows`/`cols` 使用 `parseInteger` 且校验 `rows < 0 || cols < 0` 会拒绝动态值。

模板函数签名需要 `rows=?, cols=?` 表示通用模板，需对 parser 做相同适配（`PTOTypeDefs.cpp:122-134, 202`）。

### 4.2 `MemrefToTileBuf` pass

在 InsertSync 之后将 memref 转回 tile_buf，使 Tile→Vector lowering 可以在 tile_buf 语义上工作。该 pass 需要解决：

1. **类型恢复**：从 `memref` 及其关联信息恢复 `tile_buf` 类型（rows、cols、dtype、loc、layout、fractal、pad）。
2. **语义保留**：保留 PlanMemory / InsertSync 已插入的地址规划、buffer 绑定和同步指令。

### 4.3 新增 `pto.tile_buf_addr`

从 `tile_buf` 提取 memref 的 op，是模板库和 lowering 的连接点。尚未正式定义在 `PTOOps.td` 中，需作为前置依赖实现。

### 4.4 新增 tile 属性 intrinsic 的 ODS、verifier 与 fold

在 `PTOOps.td` 中新增 `tile_rows/cols/valid_rows/valid_cols/blayout/slayout/fractal/pad` 8 个 intrinsic op，并实现：

- **Verifier**：操作数必须是 `!pto.tile_buf`；shape 类返回 `index`，配置类返回 `i32`。
- **Fold**：静态字段折叠为常量，动态字段返回空结果保留 op。

### 4.5 测试与文档

- `TileBufType` parser 的 `rows=?/cols=?` 解析测试
- tile intrinsic 的 parser / verifier / fold 测试
- 模板函数查找、克隆、特化和 inline 的单元测试
- 端到端用例（`pto.tadd` → Vector IR）
- 更新 `PTO_IR_manual.md`
