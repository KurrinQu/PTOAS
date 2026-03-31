# Tile→Vector 模板实例化机制设计

## 背景与问题

### IR 层级关系

PTOAS 编译栈中，Tile 指令是面向用户的高层抽象，向量指令是面向硬件的底层实现。我们需要用向量指令实现每一条 Tile 指令：

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

以 `pto.tadd`（逐元素加法）为例，说明两个层级之间的对应关系：

**PTO Tile IR 层**：用户编写的 tile 级别代码，`pto.tadd` 直接操作 `tile_buf`，形状和 dtype 由类型携带：

```mlir
// PTO Tile IR: 一条指令完成整个 tile 的逐元素加法
pto.tadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>)
```

**Vector IR 层**：`pto.tadd` 的向量指令实现——需要将 tile 拆解为循环，逐行逐列用 vreg 大小的向量指令处理：

```mlir
// Vector IR: pto.tadd 的向量指令实现（dtype=f32, rows=16, cols=64）
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

### 无法穷举的问题

从上面的向量实现可以看到，`dtype=f32`、`rows=16`、`cols=64` 这些值贯穿整个函数体——决定了 tile_buf 类型签名、ptr 元素类型（`ptr<f32, ub>`）、vreg 类型（`vreg<64xf32>`，其中 64 = 256B / sizeof(f32)）、以及循环边界（`%c16`、`%c64`）。

然而实际使用中 `pto.tadd` 的 tile_buf 输入千变万化——dtype 有 f16/f32/bf16 等，rows/cols 可以是任意正整数。为每种 (dtype, rows, cols) 组合手写一个向量实现是不可行的。

### 解决方案

将向量实现写成**模板函数**：
- **按 dtype 分函数**：dtype 决定 vreg 类型（如 `vreg<64xf32>` vs `vreg<128xf16>`），因此每种 dtype 一个模板函数（种类有限：f16/f32/bf16/i8/i16/i32）
- **Row/Col 用 intrinsic 占位**：定义一组 intrinsic op 从 tile_buf 类型中提取形状和配置信息，在函数体内作为占位符使用
- **Lowering pass 填充参数**：将 `pto.tadd` 替换为对模板函数的调用，填入具体的 tile_buf 类型，并折叠 intrinsic 为常量

## Tile 属性 Intrinsic Op

定义一组 intrinsic 从 `tile_buf` 中提取所有属性信息。

### 形状信息（返回 `index`）

| Intrinsic | 语义 |
|-----------|------|
| `pto.tile_rows %tb` | 提取 rows |
| `pto.tile_cols %tb` | 提取 cols |
| `pto.tile_valid_rows %tb` | 提取 v_row |
| `pto.tile_valid_cols %tb` | 提取 v_col |

### 配置信息

| Intrinsic | 语义 | 返回类型 |
|-----------|------|---------|
| `pto.tile_blayout %tb` | 提取 blayout（row_major=0/col_major=1）| `i32` |
| `pto.tile_slayout %tb` | 提取 slayout（none_box=0/row_major=1/col_major=2）| `i32` |
| `pto.tile_fractal %tb` | 提取 fractal size | `i32` |
| `pto.tile_pad %tb` | 提取 pad 策略（null=0/zero=1/max=2/min=3）| `i32` |

配置 intrinsic 返回 `i32`，值编码与 `TileBufType` 的 `getBLayoutValueI32()`/`getSLayoutValueI32()`/`getPadValueI32()` 等方法一致。

已有的 `pto.tile_buf_addr` 从 tile_buf 提取 memref。模板中输出类型为 `memref<?x?xf32, strided<[?, 1]>, #pto.address_space<vec>>`（占位）。克隆特化后，lowering pass 根据操作数的具体 tile_buf 类型自动推导并更新输出 memref 类型：

- **shape**：`rows` → dim0, `cols` → dim1
- **stride**：`blayout=row_major` → `strided<[cols, 1]>`, `blayout=col_major` → `strided<[1, rows]>`
- **address_space**：`loc` 字段映射为 `#pto.address_space<loc>`（如 `loc=vec` → `#pto.address_space<vec>`）

```
特化前: pto.tile_buf_addr %src0 : tile_buf<..., rows=?, cols=?> -> memref<?x?xf32, strided<[?, 1]>, #pto.address_space<vec>>
特化后: pto.tile_buf_addr %src0 : tile_buf<..., rows=16, cols=64, blayout=row_major, loc=vec>
        -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
```

这样向量库函数的实现中无需通过 `pto.tile_rows`/`pto.tile_cols` 显式提取行列信息来初始化 memref，特化阶段由 `tile_buf_addr` 的类型推导自动完成。

### Fold 行为与动态字段处理

每个 intrinsic op 实现 MLIR 的 `fold()` 方法，在编译期从操作数的 `TileBufType` 类型中提取常量值。`fold()` 会在 canonicalize pass 中被自动调用，无需手动编写替换逻辑。

以 `pto.tile_rows` 为例，fold 的实现逻辑：

```cpp
OpFoldResult TileRowsOp::fold(FoldAdaptor adaptor) {
  auto tbType = getOperand().getType().cast<TileBufType>();
  int64_t rows = tbType.getShape()[0];
  if (rows == ShapedType::kDynamic)
    return {};  // 动态字段，无法折叠
  // 静态字段，返回常量
  return IntegerAttr::get(IndexType::get(getContext()), rows);
}
```

工作流程：lowering pass 克隆模板函数并将 tile_buf 类型从 `rows=?` 特化为 `rows=16` 后，函数体内 `pto.tile_rows %src0` 的操作数类型变为 `tile_buf<..., rows=16, ...>`。此时调用 fold（或运行 canonicalize），`pto.tile_rows` 自动折叠为 `arith.constant 16`。

当字段为动态 `?` 时（如 `v_row=?`），intrinsic 无法折叠为常量。由于 lowering pass 直接将模板函数 inline 到调用点，未折叠的 intrinsic 会保留在调用点作用域中，后续由 canonicalize pass 在调用点上下文中解析——此时 `%src0` 的运行时信息在作用域内可达，无需额外的参数传递机制。

```mlir
// 调用点：v_row 是动态的，由上下文计算得到
%vr = ...  // 运行时计算的 valid_row
pto.tadd ins(%a: tile_buf<f32, 16, 64, v_row=?, v_col=64>, %b) outs(%c)

// Lowering inline 后（调用点处）：
%rows = arith.constant 16 : index       // rows 静态，已折叠
%cols = arith.constant 64 : index       // cols 静态，已折叠
%v_row = pto.tile_valid_rows %a : ...   // v_row 动态，保留 intrinsic
// ... 后续 canonicalize 在调用点上下文中解析 %v_row
```

### Verifier 约束

- 操作数必须是 `!pto.tile_buf` 类型
- 形状 intrinsic 返回类型必须是 `index`
- 配置 intrinsic 返回类型必须是 `i32`
- **Intrinsic 只允许出现在顶层 `pto.tile_function` 函数中**。模板函数内部可以调用辅助函数，但辅助函数不得使用 tile_buf intrinsic——所需的形状/配置信息应由顶层函数提取后作为参数传入。这避免了递归特化的复杂性，保证 lowering pass 只需特化顶层模板函数

## 模板函数结构

### 函数签名

- 函数名按 dtype 区分：`@TADD_f32`、`@TADD_f16`、`@TADD_bf16` 等
- tile_buf 参数使用动态 shape（`rows=?, cols=?`），表示通用模板
- **参数顺序与 PTO Tile IR 的 ODS 定义一致**：`ins` 参数在前，`outs` 参数在后（如 `pto.tadd` 的 `(src0, src1, dst)`）
- **参数读写属性**：每个 tile_buf 参数标注 `pto.access` 属性，声明函数体内的读写行为
  - `pto.access = "read"` — 只读（对应 `ins` 参数）
  - `pto.access = "write"` — 只写（对应 `outs` 参数）
  - `pto.access = "readwrite"` — 读写（如 in-place 操作的目标参数）
- 标记 `pto.tile_function` 属性（StringAttr），值为对应的 tile 指令名（如 `"pto.tadd"`、`"pto.tsub"`）。Lowering pass 通过此属性查找模板函数

### 模板函数存放位置

模板函数定义作为 MLIR module 内置到编译器中（编译期链接）。Tile→Vector lowering pass 在初始化时加载模板库，通过 `pto.tile_function` 属性值（如 `"pto.tadd"`）和函数签名中的 dtype 匹配模板函数。

### 示例：TADD_f32

```mlir
// 参数顺序与 pto.tadd 的 ODS 定义一致: ins(src0, src1) outs(dst)
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

  // 提取 memref（rows x cols x dtype）
  %mSrc0 = pto.tile_buf_addr %src0 : ... -> memref<?x?xf32, strided<[?, 1]>, #pto.address_space<vec>>
  %mSrc1 = pto.tile_buf_addr %src1 : ... -> memref<?x?xf32, strided<[?, 1]>, #pto.address_space<vec>>
  %mDst  = pto.tile_buf_addr %dst  : ... -> memref<?x?xf32, strided<[?, 1]>, #pto.address_space<vec>>

  // 通过 intrinsic 提取有效形状（占位符）
  %v_rows = pto.tile_valid_rows %src0 : ... -> index
  %v_cols = pto.tile_valid_cols %src0 : ... -> index
  %v_cols_i32 = arith.index_cast %v_cols : index to i32  // plt_b32 需要 i32

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index   // vector_width for f32 = 256B / 4B = 64

  pto.vecscope {
    scf.for %i = %c0 to %v_rows step %c1 {
      scf.for %j = %c0 to %v_cols step %c64 iter_args(%remain = %v_cols_i32) -> (i32) {
        // 尾部处理：remain 初始值为 valid_cols，每次迭代递减 vector_width
        // 当 remain < 64 时，plt_b32 生成部分 mask 屏蔽无效 lane
        %mask, %next = pto.plt_b32 %remain : i32 -> !pto.mask, i32

        %va = pto.vlds %mSrc0[%i, %j] : memref<?x?xf32, strided<[?, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vb = pto.vlds %mSrc1[%i, %j] : memref<?x?xf32, strided<[?, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
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


## Lowering 流程与 Pass Pipeline

### Pass Pipeline

```
┌──────────────────────────────────────┐
│  1. VF Fusion Analysis (PTO Tile IR) │
│     ● 在 tile op 粒度分析可融合性    │
│     ● 基于 tile_buf 数据流和类型兼容  │
│       性判断（ins/outs 语义）         │
│     ● 标记融合组，供后续 pass 使用    │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  2. Tile→Vector Lowering             │
│     ● 逐个 tile op 实例化对应模板    │
│     ● 克隆模板，填入具体 tile_buf    │
│       类型，折叠 intrinsic 为常量     │
│     ● 直接 inline 到调用点           │
│       （融合分析已完成，无需保留      │
│        func.call 边界）              │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  3. Fusion Pass                      │
│     ● 根据融合标记合并相邻向量循环    │
│     ● 消除中间 tile_buf 的 UB 读写   │
│     ● 减少 VF 发射次数               │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  4. Vector→LLVM Lowering             │
│     ● 向量 IR → LLVM IR             │
│     ● 由 LLVM 编译器接管后续编译      │
└──────────────────────────────────────┘
```

### 实例化与 Inline

Tile→Vector lowering pass 对每个 tile op 独立处理：克隆模板函数、填入具体 tile_buf 类型、折叠 intrinsic 为常量，然后**直接 inline 到调用点**。由于 VF 融合分析已在 PTO Tile IR 层完成，不再需要保留 `func.call` 边界。

当同一个模板被多个不同 shape 的调用点使用时，每个调用点独立克隆、特化、inline——最终结果是每个 tile op 原地替换为对应的向量循环体。

### VF Fusion 示例

VF 融合分析在 PTO Tile IR 层完成，基于 tile op 的 `ins`/`outs` 数据流和 tile_buf 类型兼容性判断：

```mlir
// PTO Tile IR 层：VF Fusion Analysis 标记融合组
pto.tadd ins(%a, %b) outs(%tmp)    // tmp = a + b
pto.tmul ins(%tmp, %d) outs(%c)    // c = tmp * d
// 分析：%tmp 是 tadd 的输出、tmul 的输入，类型兼容 → 可融合

// Tile→Vector Lowering 后：每个 tile op 已 inline 为向量循环体
// Fusion Pass 根据融合标记合并相邻循环，消除 %tmp 的 UB 读写：
//   → 减少一次 vsts + vlds，减少一次 VF 发射
```

### Lowering 详细步骤

以 `pto.tadd ins(%a, %b) outs(%c)` 为例，其中 `%a/%b/%c` 的类型为 `!pto.tile_buf<..., dtype=f32, rows=16, cols=64, ...>`：

```
Step 1: VF Fusion Analysis (PTO Tile IR)
────────────────────────────────────────
  分析 tile op 之间的数据流，标记可融合的操作组

Step 2: Tile→Vector Lowering Pass
──────────────────────────────────
输入:
  pto.tadd ins(%a, %b : tile_buf<f32, 16, 64>) outs(%c : tile_buf<f32, 16, 64>)

处理:
  1. 读取 dst (%c) 的 tile_buf 类型 → dtype=f32
  2. 查找 pto.tile_function="pto.tadd" 且 dtype=f32 的模板 → @TADD_f32
  3. 克隆 @TADD_f32，特化函数签名：tile_buf 类型具体化 (rows=16, cols=64)
  4. 折叠函数体内 intrinsic:
       - 静态字段：pto.tile_rows %a → arith.constant 16
       - 动态字段：pto.tile_valid_rows %a → 保留，inline 后在调用点上下文解析
  5. 将特化后的函数体直接 inline 到调用点，替换原 tile op

输出:
  // pto.tadd 被替换为向量循环体（已 inline，无 func.call）
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

Step 3: Fusion Pass
───────────────────
  根据 step 1 的融合标记，合并相邻向量循环，消除中间 tile_buf 的 UB 读写

Step 4: Vector→LLVM Lowering
───────────────────────────
  向量 IR → LLVM IR
```

## 讨论：Pass Pipeline 架构与 tile_buf 统一

### 问题

当前 PTOAS pipeline 中，`PTOViewToMemref` pass 将 `tile_buf` 转换为 `memref`，之后的 PlanMemory（UB 内存分配规划）和 InsertSync（管线同步插入）都基于 `memref` 数据类型工作：

```
ConvertToPTOOp → tile ops (tile_buf)
       ↓
PTOViewToMemref            ← tile_buf → memref
       ↓
PlanMemory (memref)
       ↓
InsertSync (memref)
       ↓
PTOToEmitC / PTOToA5VM (memref)
```

新增的 Tile→Vector 路径需要在 tile_buf 级别工作（intrinsic 从 tile_buf 类型提取信息），因此必须在 `PTOViewToMemref` **之前**运行。但 PlanMemory 和 InsertSync 对 Vector 路径同样必要——向量指令仍然需要 UB 内存规划和管线同步。

如果简单地在 `PTOViewToMemref` 之前分叉，Vector 路径将缺少 PlanMemory 和 InsertSync 的功能：

```
ConvertToPTOOp → tile ops (tile_buf)
       ↓
  ┌─── 路径选择 ───┐
  │                │
  ▼                ▼
PTOTileToVector    PTOViewToMemref (tile_buf → memref)
  ↓                     ↓
VF Fusion          PlanMemory (memref)    ← Vector 路径缺少这些
  ↓                     ↓
...                InsertSync (memref)    ← Vector 路径缺少这些
                        ↓
                   PTOToEmitC / A5VM
```

### 长期方向：基于 tile_buf 统一 PlanMemory 和 InsertSync

理想方案是将 PlanMemory 和 InsertSync 迁移为基于 `tile_buf` 数据类型工作，在分叉点之前运行，两条路径共享：

```
ConvertToPTOOp → tile ops (tile_buf)
       ↓
tile-level optimizations (tile_buf)
       ↓
PlanMemory (tile_buf)        ← 统一，两条路径共享
       ↓
InsertSync (tile_buf)        ← 统一，两条路径共享
       ↓
  ┌─── 路径选择 ───┐
  │                │
  ▼                ▼
PTOTileToVector    PTOViewToMemref
  ↓                     ↓
VF Fusion          PTOToEmitC / A5VM
  ↓
Inline + Cleanup
  ↓
Vector→LLVM
```

**可行性分析：**

- `tile_buf` 类型携带了 PlanMemory 和 InsertSync 所需的全部信息：shape、dtype、memory space（`loc=vec/mat/acc/...`）、config（blayout/slayout/fractal/pad）
- InsertSync 的分析逻辑本质上基于 tile op 之间的管线依赖（PIPE_V / PIPE_MTE2 / PIPE_MTE3），这些信息在 tile_buf 级别已经完备，迁移相对直接
- PlanMemory 当前依赖 `memref.alloc` / offset 计算等机制，迁移到 tile_buf 需要重新实现内存分配逻辑，改造量较大

此方向作为后续独立演进目标，不阻塞当前 Tile→Vector 路径的落地。

### 短期落地方案：InsertSync 之后 memref→tile_buf 回转

由于 PlanMemory 从 memref 迁移到 tile_buf 的改造量较大，计划与进度无法匹配，当前采用过渡方案：保留现有 PlanMemory / InsertSync 在 memref 上的实现不变，在 InsertSync 之后新增一个 `MemrefToTileBuf` pass，将 memref 转回 tile_buf，再进入 Tile→Vector lowering。

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
Tile→Vector Lowering (tile_buf)
  ↓
VF Fusion / Cleanup
  ↓
Vector→LLVM
```

**`MemrefToTileBuf` pass 的职责：**

该 pass 不重做内存规划或同步分析，而是将经过 PlanMemory / InsertSync 处理后的结果重新映射回 `tile_buf` 形式，使后续模板实例化和 tile 属性 intrinsic 仍然可以基于 `tile_buf` 类型工作。需要解决两类问题：

1. **类型恢复**：从当前 `memref` 及其关联信息恢复对应的 `tile_buf` 类型，包括 `rows`、`cols`、`dtype`、`loc`、layout、fractal、pad 等字段。
2. **语义保留**：保留 PlanMemory / InsertSync 已经插入的语义结果（地址规划、buffer 绑定、同步指令等），避免在恢复为 `tile_buf` 后丢失这些信息。

**收益：**

- 不阻塞当前 PlanMemory / InsertSync 的既有实现
- Tile→Vector lowering 仍然可以在 `tile_buf` 语义上实现，不需要被迫改写成完全基于 `memref` 的模板系统
- PlanMemory / InsertSync 的 tile_buf 化改造可以作为后续独立演进方向

## 待定事项

- **尾部 mask 处理**：当 cols 不整除 vector_width 时，使用 `pto.plt_b32` 生成 mask 屏蔽无效 lane。完整的 mask 使用示例见 `test/basic/tadd_a5.pto`（编写中）。
- **`pto.tile_buf_addr` 定义**：该 op 从 tile_buf 提取 memref，输出类型在特化阶段由 tile_buf 类型自动推导（shape/stride/address_space）。尚未正式定义在 `PTOOps.td` 中，需作为前置依赖实现。
- **`TileBufType` parser 支持 `rows=?`, `cols=?`**：当前 parser 仅 `v_row`/`v_col` 支持 `?`（`parseOptionalQuestion` → `kDynamic`），`rows`/`cols` 使用 `parseInteger` 且校验 `rows < 0 || cols < 0` 会拒绝动态值。模板函数签名需要 `rows=?, cols=?` 表示通用模板，需对 parser 做相同适配（`PTOTypeDefs.cpp:122-134, 202`）。

## Demo 文件

完整的模板 demo 见 `test/basic/tadd_a5_template.pto`，可与以下文件对比参照：
- `test/basic/tadd_a5_static.pto` — 写死版本（无 mask，无行指针递进）
- `test/basic/tadd_a5.pto` — 带 mask 的写死版本（编写中）

## Cross-Layer Impact

1. **ODS / Dialect** — `PTOTypeDefs.td` 无变更；`PTOOps.td` 新增 intrinsic op（tile_rows/cols/valid_rows/valid_cols/blayout/slayout/fractal/pad），新增 `pto.tile_function` StringAttr（值为 tile 指令名，如 `"pto.tadd"`）
2. **C++ IR** — intrinsic op 的 verifier（操作数必须是 tile_buf 类型）和 fold 实现（从 tile_buf 类型提取常量）
3. **Lowering** — 新增 Tile→Vector lowering pass（模板选择 + 克隆特化 + intrinsic 折叠 + inline）；VF Fusion Analysis 在 PTO Tile IR 层完成（lowering 之前）；Fusion Pass 在 lowering 之后合并向量循环
4. **模板库** — 模板函数定义内置到编译器中，按 (op, dtype) 索引
5. **Python bindings** — 暴露新增 intrinsic op（如需要）
6. **Tests** — intrinsic 解析/折叠测试、模板克隆/特化测试、端到端测试
7. **Docs** — 更新 `PTO_IR_manual.md`
