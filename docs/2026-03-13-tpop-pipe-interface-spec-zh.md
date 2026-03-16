# PTOAS Pipe 与 TPUSH/TPOP 对外接口规范

## 1. 文档范围

本文定义 PTOAS 中 pipe 通信相关 public PTO IR 的最终接口契约，覆盖：

- `pto.initialize_l2g2l_pipe`
- `pto.initialize_l2l_pipe`
- `pto.tpush`
- `pto.tpop`
- `pto.get_fifo_tile`
- `pto.tfree`

本文只描述对外接口语义、参数约束和使用规则，不展开编译过程中 lowering 过程。

public PTO IR 在 pipe 接口上采用以下公开数据类型：

- GM 地址使用 `!pto.ptr<elem_ty>`
- tile 数据使用 `!pto.tile_buf<...>`
- pipe handle 使用 `!pto.pipe<SrcTileType, DstTileType>`

## 2. 核心数据类型

### 2.1 `!pto.ptr<elem_ty>`

`!pto.ptr<elem_ty>` 表示 public PTO IR 中的标量 GM 指针。

特点如下：

- 用于表示 pipe 的 GM 基地址
- 不携带 shape 信息
- 可配合 `pto.addptr`、`pto.make_tensor_view` 等指令进一步构造视图

在本文定义的 pipe 接口中，`initialize_l2g2l_pipe` 的 `gm_addr` 对外使用 `!pto.ptr<elem_ty>`。

### 2.2 `!pto.tile_buf<...>`

`!pto.tile_buf<...>` 是 public PTO IR 中的 tile 语义载体，描述：

- tile 所在地址空间或角色，例如 `loc=acc/vec/mat`
- 元素类型，例如 `dtype=f32`
- 逻辑形状，例如 `rows`、`cols`
- 有效形状，例如 `v_row`、`v_col`
- 布局、分形尺寸、pad 策略等 tile 元数据

在 pipe 接口中：

- `pto.tpush` 的源 tile 使用 `!pto.tile_buf`
- `pto.get_fifo_tile` 的结果 tile 使用 `!pto.tile_buf`
- `!pto.pipe<SrcTileType, DstTileType>` 的 `SrcTileType` / `DstTileType` 对外使用 `!pto.tile_buf`

### 2.3 `!pto.pipe<SrcTileType, DstTileType>`

`!pto.pipe<SrcTileType, DstTileType>` 表示一条 pipe 的 SSA handle。

其 public 语义如下：

- 绑定一条生产者到消费者的数据通路
- 约束 `tpush` 的源 tile 语义
- 约束 `get_fifo_tile` 的结果 tile 语义

`!pto.pipe` 是函数内本地初始化并本地使用的 handle，不作为跨 kernel 函数传递的 ABI 类型。

当 producer 和 consumer 写成两个独立 kernel 函数时，双方应各自在本地重新执行对应的 `initialize_*_pipe`，而不是把 `!pto.pipe` 作为函数参数传递。

## 3. 通用接口规则

### 3.1 本地初始化原则

每个使用 pipe 的函数都应在函数内本地执行 `initialize_*_pipe`，得到本函数内的 `!pto.pipe` handle。

适用规则如下：

- 不跨函数传递 `!pto.pipe`
- producer 函数和 consumer 函数分别各自初始化
- 如果两边对应同一条逻辑 pipe，则两边初始化参数必须保持一致

### 3.2 `flag_base` 规则

`initialize_l2g2l_pipe` 和 `initialize_l2l_pipe` 都必须显式提供 `flag_base` 属性。

`flag_base` 的接口语义如下：

- 表示该 pipe 对应的同步 flag 对
- 是编译期整数字面量属性
- 由用户或前端显式指定

约束如下：

- `flag_base` 必须存在
- `flag_base` 只能取以下编译期常量之一：`0`、`2`、`4`、`6`、`8`、`10`、`12`
- 缺失 `flag_base` 时，编译报错
- 非法 `flag_base` 时，编译报错

当 producer 和 consumer 分别位于两个函数时，如果它们对应同一条逻辑 pipe，则两边必须使用相同的 `flag_base`。

### 3.3 分离 producer / consumer 时的一致性规则

若 producer 和 consumer 在不同 kernel 函数中，本规范要求两边 `initialize_*_pipe` 的以下语义参数一致：

- pipe 初始化种类一致：`initialize_l2g2l_pipe` 或 `initialize_l2l_pipe`
- `dir_mask` 一致
- `flag_base` 一致
- `SrcTileType` 一致
- `DstTileType` 一致
- `local_fifo_depth` 一致（若该接口包含该属性）

此外：

- `initialize_l2g2l_pipe` 两边应绑定同一逻辑 GM FIFO 存储
- `initialize_l2l_pipe` 两边应绑定同一逻辑 local FIFO 存储

### 3.4 执行上下文

pipe 相关操作必须位于正确的 producer / consumer 执行上下文中。

本规范不限定上下文必须通过哪一种 IR 形式表达，允许以下两类组织方式：

- 使用 `pto.section.cube` / `pto.section.vector`
- 使用带 `pto.kernel_kind` attribute 的独立 kernel 函数

## 4. Pipe 初始化接口

### 4.1 `pto.initialize_l2g2l_pipe`

`pto.initialize_l2g2l_pipe` 创建一条经 GM 中转的 pipe。

数据路径为：

`local(producer) -> GM FIFO -> local FIFO(consumer)`

#### 语法

```mlir
%pipe = pto.initialize_l2g2l_pipe {
    dir_mask = <i8>,
    flag_base = <i32>,
    local_fifo_depth = <i8>           // 可选
}
    ( %gm_addr : !pto.ptr<elem_ty>
      [, %local_addr : i32] )
    -> !pto.pipe<SrcTileType, DstTileType>
```

#### 参数

| 参数 | 类型 | 是否必须 | 说明 |
|---|---|---|---|
| `dir_mask` | 整数属性 | 必须 | 方向：`1 = C2V`，`2 = V2C` |
| `flag_base` | 整数属性 | 必须 | 该 pipe 对应的 flag 对基址 |
| `local_fifo_depth` | 整数属性 | 可选 | local FIFO 深度；未写时默认值为 `2` |
| `gm_addr` | `!pto.ptr<elem_ty>` | 必须 | GM FIFO 基地址 |
| `local_addr` | `i32` | 可选 | consumer 侧 local FIFO 基地址；未写时由上游资源规划提供 |

#### 结果

```mlir
!pto.pipe<SrcTileType, DstTileType>
```

其中：

- `SrcTileType` 对外使用 `!pto.tile_buf<...>`
- `DstTileType` 对外使用 `!pto.tile_buf<...>`

#### 语义

- producer 侧 `tpush` 将源 tile 写入 GM FIFO
- consumer 侧 `tpop` 等待 slot 就绪
- `get_fifo_tile` 暴露 consumer slot 对应的目标 tile 视图
- consumer 侧 local FIFO 深度由 `local_fifo_depth` 指定；未显式指定时默认深度为 `2`

#### 架构支持

- A3
- A5

#### 示例

```mlir
%pipe = pto.initialize_l2g2l_pipe {dir_mask = 1, flag_base = 0, local_fifo_depth = 4}
    (%gm_slot_buffer : !pto.ptr<f32>, %local_fifo_addr : i32)
    -> !pto.pipe<
         !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>,
         !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=128, v_row=32, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>>
```

### 4.2 `pto.initialize_l2l_pipe`

`pto.initialize_l2l_pipe` 创建一条 local 直传 pipe。

数据路径为：

`local(producer) -> local FIFO(consumer)`

#### 语法

```mlir
%pipe = pto.initialize_l2l_pipe {
    dir_mask = <i8>,
    flag_base = <i32>
}
    ( [%local_addr : i32] )
    -> !pto.pipe<SrcTileType, DstTileType>
```

#### 参数

| 参数 | 类型 | 是否必须 | 说明 |
|---|---|---|---|
| `dir_mask` | 整数属性 | 必须 | 方向：`1 = C2V`，`2 = V2C` |
| `flag_base` | 整数属性 | 必须 | 该 pipe 对应的 flag 对基址 |
| `local_addr` | `i32` | 可选 | local FIFO 基地址；未写时由上游资源规划提供 |

#### 结果

```mlir
!pto.pipe<SrcTileType, DstTileType>
```

其中：

- `SrcTileType` 对外使用 `!pto.tile_buf<...>`
- `DstTileType` 对外使用 `!pto.tile_buf<...>`

#### 语义

- `dir_mask = 1` 时，producer 为 Cube，consumer 为 Vector
- `dir_mask = 2` 时，producer 为 Vector，consumer 为 Cube
- 数据不经过 GM
- FIFO 深度固定为 `8`

#### 架构支持

- 仅 A5

#### 示例

```mlir
%pipe = pto.initialize_l2l_pipe {dir_mask = 1, flag_base = 0}
    (%c2v_consumer_buf : i32)
    -> !pto.pipe<
         !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>,
         !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>>
```

## 5. 数据传输接口

### 5.1 `pto.tpush`

`pto.tpush` 表示 producer 侧向 pipe 推送一份 tile。

#### 语法

```mlir
pto.tpush(%src_tile, %pipe : SrcTileType, !pto.pipe<SrcTileType, DstTileType>)
```

#### 约束

- `%src_tile` 的 public 类型使用 `!pto.tile_buf<...>`
- `%src_tile` 语义必须与 `pipe.srcTileType` 一致

### 5.2 `pto.tpop`

`pto.tpop` 等待一个 consumer slot 就绪，并返回该 slot 的逻辑编号。

#### 语法

```mlir
%slot_id = pto.tpop(%pipe : !pto.pipe<SrcTileType, DstTileType>) -> index
```

#### 结果

- `%slot_id : index`

`%slot_id` 表示一次 borrow 会话的句柄，只能和对应的 `get_fifo_tile` / `tfree` 配套使用。

### 5.3 `pto.get_fifo_tile`

`pto.get_fifo_tile` 将 `%slot_id` 映射为该 slot 对应的 borrowed FIFO tile 视图。

#### 语法

```mlir
%tile = pto.get_fifo_tile(%pipe, %slot_id
    : !pto.pipe<SrcTileType, DstTileType>, index) -> DstTileType
```

#### 结果

- 结果类型为 `DstTileType`
- public 类型使用 `!pto.tile_buf<...>`

#### 语义

- `%tile` 是 borrowed FIFO tile
- `%tile` 的生命周期截止到匹配的 `pto.tfree`

### 5.4 `pto.tfree`

`pto.tfree` 显式归还由 `pto.tpop` 借出的 consumer slot。

#### 语法

```mlir
pto.tfree(%pipe, %slot_id : !pto.pipe<SrcTileType, DstTileType>, index)
```

#### 语义

- `pto.tfree` 是必写操作
- 每个 `pto.tpop` 都必须最终对应一个 `pto.tfree`

### 5.5 生命周期与配对规则

public 接口必须满足以下配对规则：

- `slot_id` 必须来自 `pto.tpop`
- `get_fifo_tile` 与 `tfree` 必须使用与该 `tpop` 相同的 `pipe`
- 一个 borrow 会话对应一组：
  - `tpop`
  - `get_fifo_tile`
  - `tfree`
- `tfree` 必须出现在 borrowed tile 的所有使用之后

## 6. 分离 producer / consumer 的推荐写法

当 producer 和 consumer 位于两个独立 kernel 函数时，推荐写法如下：

### 6.1 Producer

```mlir
func.func @pipe_producer(%c2v_consumer_buf: i32)
    attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
  %acc_tile = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>

  %pipe = pto.initialize_l2l_pipe {dir_mask = 1, flag_base = 0}
      (%c2v_consumer_buf : i32)
      -> !pto.pipe<
           !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>,
           !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>>

  pto.tpush(%acc_tile, %pipe : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>>)
  return
}
```

### 6.2 Consumer

```mlir
func.func @pipe_consumer(%c2v_consumer_buf: i32)
    attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  %vec_tile = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>

  %pipe = pto.initialize_l2l_pipe {dir_mask = 1, flag_base = 0}
      (%c2v_consumer_buf : i32)
      -> !pto.pipe<
           !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>,
           !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>>

  %slot_id = pto.tpop(%pipe : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>>) -> index
  %fifo_tile = pto.get_fifo_tile(%pipe, %slot_id : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>>, index)
      -> !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>

  pto.tmov ins(%fifo_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%vec_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
  pto.tfree(%pipe, %slot_id : !pto.pipe<!pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>>, index)
  return
}
```

该写法的关键点是：

- 两边都本地 `initialize_l2l_pipe`
- 两边使用相同的 `flag_base`
- 两边使用相同的 pipe 语义参数
- 不跨函数传递 `!pto.pipe`

## 7. EmitC 对应关系

在 EmitC 阶段，每个 `initialize_*_pipe` 会在 init 点生成具体的 `TPipe<...>` 实例。

对外接口层面的约束如下：

- `flag_base` 直接参与具体 `TPipe<...>` 的实例化
- 编译器不再为 pipe 自动分配 `flag_base`
- 缺失 `flag_base` 必须报错

因此，对分离 producer / consumer 的两边来说，使两边生成匹配的 concrete `TPipe<...>` 的方式是：

- 各自在本地初始化
- 使用相同的 `flag_base`
- 使用相同的 pipe 语义参数

## 8. 诊断要求

以下情况属于接口级错误，应报错：

- `initialize_*_pipe` 缺失 `flag_base`
- `flag_base` 不在 `{0, 2, 4, 6, 8, 10, 12}` 中
- `initialize_l2g2l_pipe` 的 `gm_addr` 不是 `!pto.ptr<elem_ty>`
- `dir_mask` 非法
- `local_fifo_depth` 非法
- `get_fifo_tile` / `tfree` 与产生 `slot_id` 的 `tpop` 不匹配
- `tfree` 缺失
- borrowed tile 在 `tfree` 之后继续使用
