# `pto.tload` / `pto.tstore` 分析与 TileLang 模板草案设计

**日期：** 2026-04-08

## 目标

在 `lib/TileOps/` 下新增 `pto.tload` 和 `pto.tstore` 的 TileLang 模板实现，分为两部分：

1. 理想 TileLang DSL 实现下，预期写出的模板代码
2. 当前 frontend 还暂时缺失的能力

这个草案是**设计优先**，目的不是立刻让当前模板库编译通过，而是先把 5D `partition_tensor_view`
到 2D tile 的问题和缺失特性梳理清楚。

## 第一部分：`tload` / `tstore` 指令级分析

### 1. 共同搬运骨架

`TLoadVecND2ND` 和 `TStoreVecND` 是一对对偶的 ND tile 搬运例程，分别负责把 GM 上的
ND 视图装入 UB tile，以及把 UB tile 写回 GM。其核心思想不是直接对 5 维做软件五重循环，
而是把搬运拆成 **3+1 层**：

| 逻辑维度 | 承担者 | 含义 |
|---------|--------|------|
| 最内层行长 | 单次 copy 的 `lenBurst` | 每行有效字节数 |
| 次内层行数 | 单次 copy 的 `nBurst` | 连续搬运多少行 |
| 中间内层 | DMA `loop1` | 由 `set_loop1_stride_*` / `set_loop_size_*` 控制 |
| 中间外层 | DMA `loop2` | 由 `set_loop2_stride_*` / `set_loop_size_*` 控制 |
| 最外层 | CPU `for` | 软件循环更新基址 |

在原始 TileLib 实现里，单次 copy 的最内层行长由 `validCol` 表示；在本文的 TileLang
模板化设计里，PTO 的 5D `partition_tensor_view` 最后一维大小直接承接这个角色，因此后文
用 `g4` 表示模板里的逻辑最后一维，用它映射单次 copy 的 `lenBurst`。

### 2. `TLoadVecND2ND`（GM → UB）分析

#### 2.1 单次 copy 参数

`TLoadVecND2ND` 的单次 DMA copy 负责一个 `gShape3 x validCol` 的 2D 子块：

- `nBurst = gShape3`
- `lenBurst = validCol * element_bytes`
- `gmStride = gStride3 * element_bytes`
- `ubStride = TileData::Cols * element_bytes`

这里最重要的一点是：UB 侧 stride 按物理列宽 `TileData::Cols` 计算，而不是按 `validCol`
计算。也就是说，load 路径允许 `valid_shape <= physical_shape`，尾部可以是 padding 区域。

#### 2.2 UB 侧递推 stride

UB tile 按 row-major、固定行宽 `TileData::Cols` 存放，因此可以递推出：

- `dstStride2 = gShape3 * TileData::Cols`
- `dstStride1 = gShape2 * dstStride2`
- `dstStride0 = gShape1 * dstStride1`

它们分别对应：

- loop1 每推进一次，UB 目的地址跳 `dstStride2`
- loop2 每推进一次，UB 目的地址跳 `dstStride1`
- 最外层软件循环每推进一次，UB 目的地址跳 `dstStride0`

#### 2.3 DMA loop 映射

load 路径把中间两层映射为：

- `loop1 <- gShape2`
- `loop2 <- gShape1`

对应 stride 为：

- `loop1_src_stride = gStride2 * element_bytes`
- `loop1_dst_stride = dstStride2 * element_bytes`
- `loop2_src_stride = gStride1 * element_bytes`
- `loop2_dst_stride = dstStride1 * element_bytes`

然后在最外层对 `gShape0` 做软件循环，每次更新 GM/UB 基址，发一条实际的 copy 指令。

#### 2.4 清场

原始 `TLoadVecND2ND` 在 `loop1 != 1 || loop2 != 1` 时会在结尾显式恢复：

- `set_loop_size_outtoub(1, 1)`

这一步是为了避免 DMA loop 状态污染后续无关的 `copy_gm_to_ubuf`。

### 3. `TStoreVecND`（UB → GM）分析

`TStoreVecND` 与 `TLoadVecND2ND` 结构对偶，但它对 valid shape 的要求更严格。

#### 3.1 valid shape 约束

store 路径要求：

- `validCol == gShape4`
- `validRow == gShape0 * gShape1 * gShape2 * gShape3`

原因是 GM 侧写回不支持 read-modify-write。也就是说：

- load 允许把较小的 valid window 写到更大的物理 UB tile 中
- store 必须保证 UB tile 的有效窗口恰好覆盖要写回的 GM 区域

#### 3.2 DMA loop 映射

store 同样采用：

- `loop1 <- gShape2`
- `loop2 <- gShape1`

只是方向变成：

- src 是 UB
- dst 是 GM

因此 UB 侧 stride 由紧凑 row-major tile 推出，而 GM 侧 stride 直接来自 view 自身的
`gStride1 / gStride2 / gStride3`。

#### 3.3 与 load 的关键差异

和 load 相比，store 有三点需要特别记住：

1. valid shape 更严格，必须精确匹配
2. 原始 TileLib 的 store 路径没有 loop guard
3. 原始 TileLib 的 store 路径结束时没有恢复 `set_loop_size_ubtoout(1, 1)`

第三点意味着 DMA loop 状态可能泄漏到后续无关的 store。对模板化实现来说，显式补 cleanup
是更安全的选择。

### 4. 从分析到模板的映射

在本文模板化设计中，最终采用如下逻辑映射：

- `g0`：最外层软件 `for`
- `g1`：DMA `loop2`
- `g2`：DMA `loop1`
- `g3`：单次 copy 的 `n_burst`
- `g4`：单次 copy 的 `len_burst / element_bytes`

同时保持 UB 侧物理行宽始终取 `tile.shape[1]`，不取 `tile.valid_shape[1]`。

## 第二部分：TileLang 模板化实现

## 模板 API 形式

新增两个模板文件，和 `template_tadd` 并列：

- `lib/TileOps/tload_template.py`
- `lib/TileOps/tstore_template.py`

库接口签名如下：

```python
@pto.vkernel(target="a5", op="pto.tload", advanced=True)
def template_tload(src: pto.TensorView, dst: pto.Tile): ...

@pto.vkernel(target="a5", op="pto.tstore", advanced=True)
def template_tstore(src: pto.Tile, dst: pto.TensorView): ...
```

`template_tload` / `template_tstore` 是 `pto.tload` / `pto.tstore` 这两个 PTO op
的模板实现，由 PTOIR 中的 select_kernel pass 实例化。

## 草案模板代码

两个模板的完整源码同步在 `lib/TileOps/tload_template.py` 和
`lib/TileOps/tstore_template.py`，下面把当前内容内联展示：

### tload模板(ND)

```python
"""`pto.tload` 的 TileLang DSL 模板"""

import tilelang_dsl as pto

@pto.vkernel(
    target="a5",
    op="pto.tload",
    advanced=True,
)
def template_tload(src: pto.TensorView, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    g0 = src.shape[0]
    g1 = src.shape[1]
    g2 = src.shape[2]
    g3 = src.shape[3]
    g4 = src.shape[4]

    s0 = src.strides[0]
    s1 = src.strides[1]
    s2 = src.strides[2]
    s3 = src.strides[3]
    s4 = src.strides[4]

    valid_rows, valid_cols = dst.valid_shape
    ub_rows, ub_cols = dst.shape

    pto.assert_rank(src, 5)
    pto.assert_eq(s4, 1)
    pto.assert_le(valid_rows, g0 * g1 * g2 * g3)
    pto.assert_le(valid_cols, g4)
    pto.assert_le(g0 * g1 * g2 * g3, ub_rows)
    pto.assert_le(g4, ub_cols)

    n_burst = g3
    len_burst = g4 * elem_bytes
    gm_stride = s3 * elem_bytes
    ub_stride = ub_cols * elem_bytes

    dst_stride2 = g3 * ub_cols
    dst_stride1 = g2 * dst_stride2
    dst_stride0 = g1 * dst_stride1

    loop1 = g2
    loop2 = g1
    loop1_src_stride = s2 * elem_bytes
    loop1_dst_stride = dst_stride2 * elem_bytes
    loop2_src_stride = s1 * elem_bytes
    loop2_dst_stride = dst_stride1 * elem_bytes

    gm_ptr = src.as_ptr()
    ub_ptr = dst.as_ptr()

    if loop1 != 1 or loop2 != 1:
        pto.set_loop2_stride_outtoub(
            src_stride=loop2_src_stride, dst_stride=loop2_dst_stride
        )
        pto.set_loop1_stride_outtoub(
            src_stride=loop1_src_stride, dst_stride=loop1_dst_stride
        )
        pto.set_loop_size_outtoub(loop1=loop1, loop2=loop2)

    for i in range(0, g0, 1):
        src_i = pto.addptr(gm_ptr, i * s0 * elem_bytes)
        dst_i = pto.addptr(ub_ptr, i * dst_stride0 * elem_bytes)
        pto.copy_gm_to_ubuf_v2(
            dst=dst_i,
            src=src_i,
            n_burst=n_burst,
            len_burst=len_burst,
            gm_stride=gm_stride,
            ub_stride=ub_stride,
            enable_ub_pad=False,
        )

    if loop1 != 1 or loop2 != 1:
        pto.set_loop_size_outtoub(loop1=1, loop2=1)
    return
```

### tstore模板(ND)

```python
"""`pto.tstore` 的 TileLang DSL 模板"""

import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tstore",
    advanced=True,
)
def template_tstore(src: pto.Tile, dst: pto.TensorView):
    dtype = src.element_type
    elem_bytes = pto.bytewidth(dtype)

    g0 = dst.shape[0]
    g1 = dst.shape[1]
    g2 = dst.shape[2]
    g3 = dst.shape[3]
    g4 = dst.shape[4]

    s0 = dst.strides[0]
    s1 = dst.strides[1]
    s2 = dst.strides[2]
    s3 = dst.strides[3]
    s4 = dst.strides[4]

    valid_rows, valid_cols = src.valid_shape
    ub_rows, ub_cols = src.shape

    pto.assert_rank(dst, 5)
    pto.assert_eq(s4, 1)
    pto.assert_eq(valid_rows, g0 * g1 * g2 * g3)
    pto.assert_eq(valid_cols, g4)
    pto.assert_le(valid_rows, ub_rows)
    pto.assert_le(valid_cols, ub_cols)

    n_burst = g3
    len_burst = valid_cols * elem_bytes
    ub_stride = ub_cols * elem_bytes
    gm_stride = s3 * elem_bytes

    src_stride2 = g3 * ub_cols
    src_stride1 = g2 * src_stride2
    src_stride0 = g1 * src_stride1

    loop1 = g2
    loop2 = g1
    loop1_src_stride = src_stride2 * elem_bytes
    loop1_dst_stride = s2 * elem_bytes
    loop2_src_stride = src_stride1 * elem_bytes
    loop2_dst_stride = s1 * elem_bytes

    ub_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()

    if loop1 != 1 or loop2 != 1:
        pto.set_loop2_stride_ubtoout(
            src_stride=loop2_src_stride, dst_stride=loop2_dst_stride
        )
        pto.set_loop1_stride_ubtoout(
            src_stride=loop1_src_stride, dst_stride=loop1_dst_stride
        )
        pto.set_loop_size_ubtoout(loop1=loop1, loop2=loop2)

    for i in range(0, g0, 1):
        src_i = pto.addptr(ub_ptr, i * src_stride0 * elem_bytes)
        dst_i = pto.addptr(gm_ptr, i * s0 * elem_bytes)
        pto.copy_ubuf_to_gm_v2(
            dst=dst_i,
            src=src_i,
            n_burst=n_burst,
            len_burst=len_burst,
            gm_stride=gm_stride,
            ub_stride=ub_stride,
        )

    if loop1 != 1 or loop2 != 1:
        pto.set_loop_size_ubtoout(loop1=1, loop2=1)
    return
```

## 目标 lowering 骨架

### 共同骨架

两个模板都编码分析文档中的 **3+1 层循环分解**：

| 维度 | 承担者 |
|------|--------|
| `g4`（lenBurst 方向） | 单次 copy 的 `len_burst` |
| `g3`（nBurst 方向） | 单次 copy 的 `n_burst` |
| `g2` | DMA 硬件 loop1 |
| `g1` | DMA 硬件 loop2 |
| `g0` | CPU 软件 `for` |

共享参数：

- `n_burst = g3`
- `len_burst = g4 * element_bytes`
- `gm_stride = s3 * element_bytes`
- `ub_stride = ub_cols * element_bytes`（注意是 `tile.shape[1]`，不是 `valid_shape[1]`）
- UB 侧紧凑 row-major 递推：`dst_stride2 = g3 * ub_cols`、`dst_stride1 = g2 * dst_stride2`、
  `dst_stride0 = g1 * dst_stride1`

### `template_tload`（GM → UB）

参考分析文档 §1 和 `pto-isa/include/pto/npu/a5/TLoad.hpp` 的 `TLoadVecND2ND`。

实现草案见 `lib/TileOps/tload_template.py`，核心要点：

- `loop1 = g2`，`loop2 = g1`；loop 守卫 `if loop1 != 1 or loop2 != 1` 保持与 C++ 一致
- 最外层软件 `for i in range(0, g0, 1)`：每次迭代更新 `i * s0 * elem_bytes` 和
  `i * dst_stride0 * elem_bytes`
- 结尾恢复 `set_loop_size_outtoub(1, 1)`
- 断言**宽松**：`valid_rows ≤ g0*g1*g2*g3`，`valid_cols ≤ g4`（允许 padding 尾部）

### `template_tstore`（UB → GM）

参考分析文档 §2 和 `pto-isa/include/pto/npu/a5/TStore.hpp` 的 `TStoreVecND`。

实现草案见 `lib/TileOps/tstore_template.py`，核心要点：

- 和 tload 结构对偶，但寄存器族换成 `*_ubtoout`
- 断言**严格**：`valid_rows == g0*g1*g2*g3`，`valid_cols == g4`
  （DMA 不支持对 GM 的 read-modify-write，必须整块写入）
- 原始 TileLib 的 `TStoreVecND` 没有 loop 守卫，也不在函数结束时复位；本草案模板
  **主动补齐** 这两点，避免 DMA 寄存器状态污染后续无关的 store（分析文档附录第 3 项）

两个模板的完整代码见 `lib/TileOps/tload_template.py` 和 `lib/TileOps/tstore_template.py`。

## 第三部分：缺失特性与待补能力

### 缺失特性一览

这里先给出一个汇总，后续小节对每一项展开论证。
除了上一版草案已提到的 rank-5 TensorView / strides / bytewidth / assert*
之外，本次 review 新识别出了四个**更关键、不补无法真正走通模板**的 gap。

| # | 缺失项 | 类别 | 重要性 |
|---|--------|------|--------|
| 1 | rank-5 `TensorView.shape` / `.strides` | 类型系统 | blocking |
| 2 | rank-5 `partition_tensor_view` 的 authoring/lowering 路径 | IR pipeline | blocking |
| 3 | `pto.copy_gm_to_ubuf` / `pto.copy_ubuf_to_gm` 的 DSL Python wrapper 暴露 burst 字段 | DSL frontend wrapper | blocking |
| 4 | `TensorView.as_ptr()` | 类型系统 | blocking |
| 5 | `pto.bytewidth(dtype)` | scalar builtin | 必要 |
| 6 | `pto.set_loop*_stride_*` / `pto.set_loop_size_*` 的 src/dst 与 loop1/loop2 **命名参数** | DMA intrinsic | 可暂缓（ergonomics） |
| 7 | `pto.assert_rank / assert_eq / assert_le`（实例化期断言） | frontend 约束 | 可暂缓 |
| 8 | fp4 偏移减半的 dtype 特化分支 | dtype 处理 | 可暂缓 |
| 9 | 非 row-major UB tile stride 泛化 | tile 布局 | 可暂缓 |

### 需要补齐的特性（展开论证）

### 1. rank-5 `TensorView.shape` / `.strides`

当前 TileLang frontend 的 TensorView 主要围绕 rank-2 工作
（`tilelang-dsl-guide.md` §TensorView Types: "The current stable DMA-oriented profile is rank-2 only"），
而 `pto.tload` / `pto.tstore` 消费的是 **5D `partition_tensor_view`**。需要补齐：

- `src.shape[0..4]` / `src.strides[0..4]` 的访问
- rank-5 TensorView 作为 `@pto.vkernel` 参数
- rank-5 shape/stride 的语义检查与 lowering 传递

没有这一项，后面所有讨论都不成立。

### 2. rank-5 `partition_tensor_view` 的 authoring/lowering

PTO IR 已能表达 rank-5 `partition_view`，但 TileLang DSL 前端尚未支持
从 DSL 这一侧构造/接收这种 view。需要打通：

- 前端参数/类型建模
- 静态 + 动态 shape/stride 两种场景
- pointer materialization（和 §4 一起）

### 3. `pto.copy_gm_to_ubuf` / `pto.copy_ubuf_to_gm` 的 DSL Python wrapper 暴露 burst 字段

**这一条的本质是 DSL frontend wrapper 与 IR 接口不一致，不是 IR 缺能力。**

在 VPTO IR 一级，`pto.copy_gm_to_ubuf` 和 `pto.copy_ubuf_to_gm` **本来就是 burst-mode**
的（见 `docs/release/vpto-spec-v0.2.md` §`pto.copy_gm_to_ubuf` / §`pto.copy_ubuf_to_gm`，
约 1240 / 1270 行）。IR operands 表如下：

```mlir
pto.copy_gm_to_ubuf %gm_src, %ub_dst,
    %sid, %n_burst, %len_burst, %left_padding, %right_padding,
    %src_stride, %dst_stride, ...

pto.copy_ubuf_to_gm %ub_src, %gm_dst,
    %sid, %n_burst, %len_burst, %reserved, %dst_stride, %src_stride
```

这与 C++ `copy_gm_to_ubuf_align_v2` / `copy_ubuf_to_gm_align_v2` 是一一对应的，
`%n_burst / %len_burst / %src_stride / %dst_stride` 完整暴露。

**问题出在 TileLang DSL 的 Python wrapper**：当前 `tilelang-dsl-guide.md` 把同一个
op 暴露成了一个 *不带 burst 字段* 的窄签名：

```
pto.copy_gm_to_ubuf(src: GMPtr, dst: UBPtr,
                   src_offset, src_stride0, src_stride1,
                   dst_offset, dst_stride0,
                   transpose, pad_left, pad_right, pad_value) -> None
pto.copy_ubuf_to_gm(src: UBPtr, dst: GMPtr,
                    src_offset, src_stride0, src_stride1,
                    dst_offset, dst_stride0, dst_stride1) -> None
```

参数表里 **完全没有** `n_burst / len_burst`，因此模板从 DSL 这一侧无法表达
`TLoadVecND2ND` / `TStoreVecND` 最内 2 层 `nBurst × lenBurst` 的折叠拷贝。
不补齐这一条，模板只能在 IR 层手写，无法走 TileLang 路径。

**建议方案：** 修正 DSL Python wrapper，让它直接对齐 IR operands：

```python
pto.copy_gm_to_ubuf(
    dst: UBPtr, src: GMPtr,
    n_burst: pto.i32,         # gShape3
    len_burst: pto.i32,       # gShape4 * element_bytes
    src_stride: pto.i64,      # gStride3 * element_bytes (源行间距)
    dst_stride: pto.i64,      # tile.shape[1] * element_bytes (目的行间距)
    sid: pto.i32 = 0,
    left_padding: pto.i32 = 0,
    right_padding: pto.i32 = 0,
    pad_value: pto.i64 = 0,
) -> None

pto.copy_ubuf_to_gm(
    dst: GMPtr, src: UBPtr,
    n_burst: pto.i32,
    len_burst: pto.i32,
    src_stride: pto.i64,
    dst_stride: pto.i64,
    sid: pto.i32 = 0,
) -> None
```

修复范围只在 `tilelang-dsl/python/tilelang_dsl/` 的 wrapper 层（以及对应文档），
**不需要新增 IR op**，也不需要 `_v2` 命名共存——直接用现有 op 名即可，原有的
*不带 burst* 的签名应当作 frontend bug 修正掉。

> 注：上面草案模板用的 `pto.copy_gm_to_ubuf_v2` / `pto.copy_ubuf_to_gm_v2` 名字
> 只是为了在 review 阶段视觉上区分「这是新签名」，正式实现时直接复用 op 原名
> `pto.copy_gm_to_ubuf` / `pto.copy_ubuf_to_gm`。

### 4. `TensorView.as_ptr()`

当前 DSL 已经提供 `Tile.as_ptr() -> UBPtr`，但 **没有** 对 TensorView 的对偶接口。
模板里需要从 5D TensorView 拿到 GM 基指针，再由软件循环用 `pto.addptr` 推进。

**建议补充方案：**

```python
class TensorView:
    def as_ptr(self) -> GMPtr: ...     # 对应已有的 Tile.as_ptr()
```

### 5. `pto.bytewidth(dtype)`

模板内部需要统一把元素个数换算为字节数：

- `len_burst = g4 * elem_bytes`
- `gm_stride = s3 * elem_bytes`
- `ub_stride = ub_cols * elem_bytes`

需要一个明确的 frontend 内建 `pto.bytewidth(dtype) -> pto.i32`，在实例化阶段可求值
（对 fp4 要处理 bit-width 不满 1 字节的 case，或先作为 error 拦截）。

### 6. `set_loop*_stride_*` / `set_loop_size_*` 的命名参数（可暂缓）

当前 DSL 签名是：

```
pto.set_loop2_stride_outtoub(stride0: pto.i64, stride1: pto.i64)
pto.set_loop1_stride_outtoub(stride0: pto.i64, stride1: pto.i64)
pto.set_loop_size_outtoub(size0: pto.i64, size1: pto.i64)
```

硬件端这些寄存器都是 **单个 u64 打包两个字段**：

- stride 寄存器：`[39:0] = src_byte_stride`，`[60:40] = dst_byte_stride`
- size 寄存器：`[20:0] = loop1 count`，`[40:21] = loop2 count`

DSL 用 `(stride0, stride1)` / `(size0, size1)` 这种 **位置参数** 让调用方必须
记忆哪个位是 src、哪个位是 dst、哪个是 loop1、哪个是 loop2，
容易写错且无法静态校验。模板可以考虑这样表达：

```python
pto.set_loop2_stride_outtoub(src_stride=..., dst_stride=...)
pto.set_loop1_stride_outtoub(src_stride=..., dst_stride=...)
pto.set_loop_size_outtoub(loop1=..., loop2=...)
```

**建议：** DSL 这几个内建改成 keyword-only 参数，让语义在 authoring 层就显式固定；
lowering 时再按硬件的 bit field 约定打包。这不是新能力，只是给现有 intrinsic 加
命名参数 alias，成本低、收益大。

不过这一条不阻塞模板主路径：现有的 `(stride0, stride1)` / `(size0, size1)` 位置参数
仍可调用，只要在调用点用注释固定 src/dst、loop1/loop2 的位置约定即可。先暂缓，
等模板主路径走通后再补齐。

### 7. 实例化期断言 `pto.assert_`（可暂缓）

模板使用：

- `pto.assert_rank(view, 5)`
- `pto.assert_eq(a, b)` / `pto.assert_le(a, b)`

这些不是运行时控制流，而是 **模板实例化期** 或 **前端语义阶段** 的约束检查。
需要支持：

- rank 检查
- shape / valid_shape 代数等式与不等式
- innermost stride 是否为 1
- load 用 `assert_le`，store 用 `assert_eq`（见分析文档 §2.1：load/store 对 valid
  shape 的约束是不对称的）

这一条先暂缓：在补齐之前，相关约束可以靠调用方契约 + 实例化期的隐式形状匹配维持，
断言只是显式化检查，不阻塞模板主路径跑通。

### 8. fp4 特化（可暂缓）

C++ 源码里 `gStride0 / dstStride0 / srcStride0 >>= 1`（`float4_e*x2` 一个元素 4 bit，
硬件按 b8 粒度搬运）。草案模板暂不处理 fp4，`pto.bytewidth(fp4)` 应直接报错拦截。
后续补齐时，需要在 DSL 侧支持 **dtype 分支的实例化期 if**，或在 `pto.bytewidth` 的
返回值上使用分数字节 + 最外层偏移预处理。

### 9. 非 row-major UB tile stride（可暂缓）

草案假定 UB tile 是紧凑 row-major、行宽 `tile.shape[1]`。非 row-major tile location
（例如 `mat` / `acc`）暂不在这次草案范围。
