# Tile Lib 向量库方案设计

## 第一章 背景与问题

### 1.1 当前编译栈与编译时长问题

当前从 DSL 到硬件二进制的完整编译栈如下：

```
PTO DSL (TileLang...)
       ↓
     PTOAS (MLIR)
       ↓
  Tile Lib (CCE)          ← C++ 模板库
       ↓
     CCEC                 ← C++ 编译器
       ↓
    LLVM IR
       ↓
    BiSheng
       ↓
  Davinci Binary
```

这条编译栈层次较深。PTOAS 生成 CCE C++ 代码后，需要经过 C++ 模板实例化和 CCEC 编译才能产生 LLVM IR，再由 BiSheng 编译器生成最终的 Davinci 二进制。其中 **C++ 模板实例化和 CCE 编译** 是主要的编译时间瓶颈。

我们希望简化编译栈，跳过 CCE 代码生成和编译的过程，直接从 PTOAS 输出 LLVM IR：

```
PTO DSL (TileLang...) + Tile Lib
       ↓
     PTOAS (MLIR)         ← 直接输出 LLVM IR，跳过 CCE
       ↓
    LLVM IR
       ↓
    BiSheng
       ↓
  Davinci Binary
```

这样可以显著缩短编译时间。但当前的 Tile Lib 是基于 CCE 和 C++ 模板开发的，因此需要用其它方式重新实现 Tile Lib。

### 1.2 PTOAS 中向量库实现的挑战

PTOAS 中目前设计两层粒度的 IR：

- **PTO Tile IR**：面向上层用户的高层抽象，操作对象是 `tile_buf`，一条指令表达完整的 tile 语义（如 `pto.tadd`、`pto.tmul`、`pto.tload`）。
- **Vector IR (vPTO)**：面向底层硬件的指令接口，操作对象是 `vreg`/`ptr`，需要显式循环、显式寄存器宽度、显式 mask 处理（如 `pto.vadd`、`pto.vlds`、`pto.vsts`）。

Tile Lib 的一种实现方式是直接使用 Vector IR 编写。以 `pto.tadd`（逐元素加法）为例，在 Tile IR 层只需一条指令：

```mlir
pto.tadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>)
```

而用 Vector IR 实现同样的语义（`dtype=f32, rows=16, cols=64`），需要展开为完整的向量循环：

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

直接基于 Vector IR 开发 Tile Lib 面临以下困难：

1. **MLIR 语法门槛高**：需要熟悉 `memref`、`index`、`strided` 等 MLIR 数据类型和语法，使用 MLIR 的方式定义变量、表达运算和控制流。
2. **参数组合无法穷举**：`dtype` 有 f16/f32/bf16 等，`rows`/`cols` 可以是任意正整数，为每种 `(op, dtype, rows, cols, layout)` 组合手写向量实现不可行。

因此，直接基于 PTO Vector IR 开发 Tile Lib，技术难度大且工作无法收敛。

## 第二章 方案：使用 Python 开发 Tile Lib

### 2.1 总体思路

为了降低开发门槛并解决参数组合的穷举问题，我们采用 PTO DSL 来编写 Tile Lib 的向量库实现。这套语法定义在 TileLang 中，库开发者使用 Python 编写模板函数，由 PTOAS 编译器在编译时进行实例化。

整体方案：

1. **用 Python DSL 编写模板函数**：使用 `pto.Tile` 数据类型和向量操作接口，按 Tile 指令语义编写向量实现。
2. **编译器实例化模板**：PTOAS 在编译过程中遇到 Tile op 时，调用对应的模板函数，填入具体的 `tile_buf` 类型参数，生成特化后的向量 IR。
3. **inline 到调用点**：特化后的向量 IR 直接 inline 到原 Tile op 的位置，继续后续优化和 lowering 流程。

### 2.2 TADD 模板示例

以 `pto.tadd`（逐元素加法）为例，使用 Python DSL 编写的模板函数如下：

```python
@pto.tile_template(target="a5", op="pto.tadd")
def template_tadd(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = src0.element_type
    elem_size = src0.element_size
    rows, cols = src0.shape
    v_rows, v_cols = src0.valid_shape

    for i in range(0, v_rows, 1):
        remaining = v_cols
        for j in range(0, v_cols, 256 / elem_size):
            all_mask, remaining = pto.make_mask(dtype, remaining)
            vec_a = pto.vlds(a[i, j])
            vec_b = pto.vlds(b[i, j])
            result = pto.vadd(vec_a, vec_b, all_mask)
            pto.vsts(result, c[i, j], all_mask)
```

代码解读：

- **`@pto.tile_template`** 装饰器指示这是一个 `pto.tadd` 指令的模板，会在编译时进行实例化。
- **输入参数**为 3 个 `pto.Tile` 数据类型参数，2 个输入（`src0`、`src1`），1 个输出（`dst`）。
- 通过 **`Tile` 数据类型接口**获取元素类型（`element_type`）、元素大小（`element_size`）、静态 shape（`shape`）和 valid shape（`valid_shape`）信息。
- 通过 **2 层循环**分别遍历 tile 的行和列。
- 通过 **`pto.make_mask`** 指令，根据基础数据类型大小及有效数据数量设置 mask 寄存器。
- 通过 **`pto.vlds`** 指令，以 `a[i, j]` 和 `b[i, j]` 为起始地址分别将数据读入向量寄存器。
- 通过 **`pto.vadd`** 计算相加结果，写入寄存器 `result`。
- 通过 **`pto.vsts`** 将 `result` 写入以 `c[i, j]` 为起始的地址区间。

### 2.3 TileLang DSL 语法参考

#### 2.3.1 基础数据类型

| DSL 类型 | 说明 | 位宽 |
|----------|------|------|
| `pto.i8` | 8 位整数 | 8 |
| `pto.i16` | 16 位整数 | 16 |
| `pto.i32` | 32 位整数 | 32 |
| `pto.i64` | 64 位整数 | 64 |
| `pto.f16` | 半精度浮点 | 16 |
| `pto.bf16` | BFloat16 | 16 |
| `pto.f32` | 单精度浮点 | 32 |

Python 字面量自动推导类型：`int` → `pto.i32`，`float` → `pto.f32`。

#### 2.3.2 Tile 数据类型

`pto.Tile` 表示一个带有布局和配置信息的数据块，对应 MLIR 中的 `!pto.tile_buf` 类型。

**Tile 属性接口：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `shape` | `tuple[int, ...]` | Tile 的完整维度（rows, cols） |
| `valid_shape` | `tuple[int, ...]` | 有效数据维度（v_row, v_col），可能小于 shape |
| `element_type` | `Type` | 元素数据类型（如 `pto.f32`） |
| `element_size` | `int` | 元素字节大小（如 f32 → 4） |
| `memory_space` | `MemorySpace` | 内存空间（GM, UB） |
| `config` | `TileConfig` | 布局和 padding 配置 |

**Tile 配置：**

```python
pto.BLayout.ROW_MAJOR     # 行主序
pto.BLayout.COL_MAJOR     # 列主序
pto.SLayout.NONE_BOX      # 无二级布局
pto.PadValue.NULL          # 无 padding
pto.PadValue.ZERO          # 零填充
```

#### 2.3.3 向量操作接口

向量寄存器固定 256 字节宽度，每次处理的元素数量由数据类型决定：f32 → 64 个元素，f16 → 128 个元素。

**Mask 操作：**

| 操作 | 说明 |
|------|------|
| `pto.make_mask(dtype, remaining)` | 根据数据类型和剩余元素数量生成 mask，返回 `(mask, new_remaining)` |
| `pto.make_mask(dtype, PAT.ALL)` | 生成全 1 mask |

**向量 Load/Store：**

| 操作 | 说明 |
|------|------|
| `pto.vlds(tile[i, j])` | 从 Tile 的 `[i, j]` 位置加载一个向量寄存器的数据 |
| `pto.vsts(vec, tile[i, j], mask)` | 将向量寄存器数据写入 Tile 的 `[i, j]` 位置 |

**二元向量运算：**

| 操作 | 说明 |
|------|------|
| `pto.vadd(vec1, vec2, mask)` | 逐元素加法 |
| `pto.vsub(vec1, vec2, mask)` | 逐元素减法 |
| `pto.vmul(vec1, vec2, mask)` | 逐元素乘法 |
| `pto.vdiv(vec1, vec2, mask)` | 逐元素除法 |
| `pto.vmax(vec1, vec2, mask)` | 逐元素取大 |
| `pto.vmin(vec1, vec2, mask)` | 逐元素取小 |

**一元向量运算：**

| 操作 | 说明 |
|------|------|
| `pto.vabs(vec, mask)` | 逐元素绝对值 |
| `pto.vexp(vec, mask)` | 逐元素指数 |
| `pto.vln(vec, mask)` | 逐元素对数 |
| `pto.vsqrt(vec, mask)` | 逐元素开方 |
| `pto.vrelu(vec, mask)` | 逐元素 ReLU |

**向量-标量运算：**

| 操作 | 说明 |
|------|------|
| `pto.vmuls(vec, scalar, mask)` | 向量乘标量 |
| `pto.vadds(vec, scalar, mask)` | 向量加标量 |

#### 2.3.4 控制流

**循环**使用 Python 的 `range` 语法：

```python
for i in range(0, v_rows, 1):
    # 循环体
```

当循环边界来自 `shape`（编译期常量）时，DSL 在 Python 层展开循环；当来自 `valid_shape`（可能是运行时动态值）时，生成 `scf.for` MLIR 循环。

## 第三章 PTOAS 编译器：TileOp Expand

### 3.1 编译流程

PTOAS 编译器的输入可以是 Tile 指令、向量指令、或两者的混合。完整的编译 pipeline 如下：

```
输入：TileOp / 向量指令 / TileOp + 向量指令混合
       ↓
  VF Fusion Analysis        ← 在 Tile IR 层分析可融合的操作组
       ↓
  PlanMemory                ← UB 内存分配规划
       ↓
  InsertSync                ← 管线同步插入
       ↓
  Expand TileOp             ← 将 TileOp 实例化为向量指令
       ↓
  VF Fusion                 ← 合并相邻向量循环，消除中间 UB 读写
       ↓
  LLVM IR
```

其中 **Expand TileOp** 是本方案的核心 pass，负责将 Tile 指令展开为实例化后的向量库指令。

### 3.2 Expand TileOp Pass 的工作流程

以编译时遇到 `pto.tadd` 为例，Expand TileOp pass 的处理步骤如下：

```
Step 1: 识别 Tile Op
───────────────────
  遇到 pto.tadd ins(%a, %b) outs(%c)
  从操作数的 tile_buf 类型提取属性：
    dtype=f32, rows=16, cols=64, v_row=16, v_col=64,
    blayout=row_major, slayout=none_box, fractal=512, pad=0

Step 2: 匹配模板函数
───────────────────
  根据 Tile op 种类（pto.tadd）和 dtype（f32）
  查找对应的 Python DSL 模板 → template_tadd

Step 3: 实例化模板
───────────────────
  调用模板函数，填入具体的 tile_buf 类型参数
  Python DSL 在编译期折叠静态字段（rows, cols, elem_size, ...）
  动态字段（v_row, v_col）生成为函数参数
  输出实例化后的 MLIR 向量 IR

Step 4: Inline 到调用点
───────────────────────
  将实例化后的向量 IR 直接 inline 到原 pto.tadd 的位置
  绑定函数参数到调用点的实际值
  删除原 Tile op

Step 5: Cleanup
───────────────
  运行 canonicalize，消除多余常量和中间符号
```

### 3.3 Pass 输出示例

经过 Expand TileOp 处理后，原来的 `pto.tadd` 被替换为向量循环体：

**输入（Tile IR）：**

```mlir
pto.tadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>)
```

**输出（Vector IR）：**

```mlir
func.func @TADD_f32(
    %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=?, v_col=?,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=?, v_col=?,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>,
    %dst:  !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=?, v_col=?,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>)
  {

  // 1. 从 tile_buf 提取 memref（shape 和 stride 为占位，特化后推导）
  %mSrc0 = pto.tile_buf_addr %src0 : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %mSrc1 = pto.tile_buf_addr %src1 : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>
  %mDst  = pto.tile_buf_addr %dst  : ... -> memref<16x64xf32, strided<[64, 1]>, #pto.address_space<vec>>

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
            : memref<16x64xf32, strided<[?, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vb = pto.vlds %mSrc1[%i, %j]
            : memref<16x64xf32, strided<[?, 1]>, #pto.address_space<vec>> -> !pto.vreg<64xf32>
        %vc = pto.vadd %va, %vb, %mask
            : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
        pto.vsts %vc, %mDst[%i, %j], %mask
            : !pto.vreg<64xf32>, memref<16x64xf32, strided<[?, 1]>, #pto.address_space<vec>>, !pto.mask

        scf.yield %next : i32
      }
    }
  }
  return
}
```


## 第四章 前置工作

### 4.1 Python DSL 扩展

| 工作项 | 说明 |
|--------|------|
| `@pto.tile_template` 装饰器 | 标记模板函数，指定对应的 Tile op 和 target |
| `pto.Tile` 属性接口 | 支持 `shape`、`valid_shape`、`element_type`、`element_size` 等属性访问 |
| `Tile` 下标访问 | 支持 `tile[i, j]` 语法用于 `vlds`/`vsts` 的地址计算 |
| 动态循环边界 | 当 `valid_shape` 为运行时动态值时，`range` 生成 `scf.for` |

### 4.2 PTOAS 编译器：Expand TileOp Pass

| 工作项 | 说明 |
|--------|------|
| 模板查找机制 | 根据 Tile op 种类和 dtype 匹配 Python DSL 模板 |
| 模板实例化 | 调用 Python DSL，传入具体 `tile_buf` 类型，获取实例化后的 MLIR |
| MLIR 解析与 inline | 解析生成的 MLIR 文本，inline 到调用点，绑定参数 |
| Cleanup | 实例化后运行 canonicalize 清理冗余 |

### 4.3 测试与文档

- Python DSL 模板编写和实例化的单元测试
- Expand TileOp pass 的端到端测试（`pto.tadd` → Vector IR）
- 融合场景测试（多个 Tile op 连续使用后的 VF Fusion）
- 更新 `PTO_IR_manual.md` 和 TileLang DSL Guide
