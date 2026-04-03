# VPTO `pto.vkernel` 模板化与 JIT 绑定计划

## 背景

后续 DSL 需要支持 tile 级实现的模板化编译，但这部分不应继续污染当前第一阶段 `@pto.vkernel` 基础语义与 lowering 主线。

当前计划将模板化问题独立出来，单独约束：

- caller 如何构造复合参数对象
- JIT 绑定时如何拆分静态/动态值
- 哪些值进入 specialization key
- 哪些值进入最终 runtime ABI

## 当前结论

首版不做通用静态/动态参数系统，不先设计通用 `argstruct` 或 `static_type()`。

首版只引入两个概念：

- `@pto.struct`
- `pto.const(...)`

其中：

- `@pto.struct` 用来定义 DSL 可识别的参数结构类型
- `pto.const(...)` 是通用静态绑定标记

## `@pto.struct` 首版模型

首版机制层使用通用 `@pto.struct`。

在库层，我们可以内置一个标准参数结构 `Tile`；但它不是语法特例，而是通过同一套 `@pto.struct` 机制定义出来的普通 DSL 参数结构。

首版先用这个内置 `Tile` 结构来承载：

- `ub_ptr`
- `shape`

示意：

```python
@pto.struct
class Tile:
    ub_ptr: pto.ptr
    shape: pto.const
```

这意味着两件事：

- `pto.Tile` 只是库里预置好的一个 `@pto.struct`
- 用户后续若需要别的参数结构，也可以用同样写法自行定义普通 class

例如：

```python
@pto.struct
class Block:
    ub_ptr: pto.ptr
    shape: pto.const
```

这里 `Block` 和 `Tile` 在机制层没有区别，只是库层约定和命名不同。

约定：

- `ub_ptr` 表示一个动态指针值字段；其中类型对象参数进入模板化维度，内存空间仍属于静态指针类型对象的一部分，不参与 runtime 传值
- `shape` 表示一个 `pto.const(...)` 静态值字段
- `Tile` 只是库内置的一个标准结构，不是机制层保留关键字
- 后续若增加别的参数结构，或用户自己定义新的参数结构，也沿用同一套 `@pto.struct` + `pto.const(...)` 语义，而不是为某个结构单独硬编码机制

## 固定设计: `@pto.struct` 与 `pto.ptr`

这一版先把 `@pto.struct` 和 `pto.ptr` 的语义固定下来，后续模板化/JIT 绑定都以这组约束为准。

### `pto.ptr`

本文档统一使用下面这组术语：

- `pto.f32` / `pto.f16` / `pto.i32`
  - 叫“类型对象”
- `pto.ptr(pto.f32, "ub")`
  - 叫“指针类型对象”

因此，`pto.ptr(elem_type, memory_space)` 可以理解为“由类型对象和内存空间标签构成的指针类型对象”。

`pto.ptr(elem_type, memory_space)` 分成两层：

- 动态部分:
  - 指针值本身
- 静态部分:
  - `elem_type`
  - `memory_space`

也就是说：

- `!pto.ptr<f32, ub>` 中，`f32` 对应类型对象，`ub` 对应内存空间标签；二者都属于静态类型层
- runtime ABI 里只需要传这根 pointer 的动态值
- 任何依赖元素类型的信息，例如元素字节数、vector step、访存解释，都应从静态指针类型对象推导，而不是作为额外动态参数传入
- 在这版模板化设计里，`elem_type` 对应的类型对象进入模板化维度，可以在 `.jit(...)` 时参与实例化；`memory_space` 暂不进入模板化维度，仍由 DSL 函数签名中的 `pto.ptr(..., memory_space)` 静态确定

### `@pto.struct`

`@pto.struct` 用来把一个普通 Python class 声明提升成 DSL 可识别的参数结构类型。

因此：

- 机制层提供的是 `@pto.struct`
- 库层可以内置若干标准结构，例如 `Tile`
- 用户也可以按同样方式自定义别的参数结构 class

首版先以：

```python
@pto.struct
class Tile:
    ub_ptr: pto.ptr
    shape: pto.const
```

作为模板化样例。

其中：

- `ub_ptr` 的字段类型固定为某个指针类型对象 `pto.ptr(...)`
- `ub_ptr` 里只有指针值动态；其中类型对象参数可模板化，`memory_space` 静态
- `shape` 表达结构上的静态元数据
- `shape` 首版通过 `pto.const(...)` 绑定为静态值
- 因此这版模板参数包含两部分：
  - 类型对象参数：`ub_ptr` 的 `elem_type`
    这里 `elem_type` 只是字段名，其字段值属于“类型对象参数”
  - `pto.const(...)` 静态值参数：`shape` 等显式静态绑定值

### 与模板化 `abs` 用例的对应关系

下文的模板化 `abs` 用例按这组固定设计来写：

- `src.ub_ptr` / `dst.ub_ptr`
  - 对应 `@pto.struct` 参数对象中的动态 pointer value 字段
- `src.shape[0] * src.shape[1]`
  - 对应从 tile 静态 `shape` 推导总元素数
- `256 // src.ub_ptr.elem_bytes`
  - 对应从 `pto.ptr` 的模板化类型对象参数推导 vector step
- `.jit(src=src_tile, dst=dst_tile)`
  - 对应 caller 绑定类型对象参数和 `pto.const(...)` 静态值参数，共同决定实例化结果；不需要再次绑定 `memory_space`

## `pto.const(...)` 绑定语义

`pto.const(...)` 不是普通运行时构造 op，而是 JIT 绑定阶段的静态值标记。

在当前统一口径下，`pto.const(...)` 只负责模板参数中的“静态值参数”这一半；与之并列的另一半是类型对象参数。

语义约束：

- 被 `pto.const(...)` 包裹的值，在 `kernel.jit(...)` 绑定后被冻结为静态值
- 这些静态值进入 specialization key
- 这些静态值不会作为参数传入最终函数 ABI
- lowering 时它们应按字面量/常量直接处理
- 未被 `pto.const(...)` 标记的部分默认保留为动态值
- 对 `pto.ptr` 而言，只有指针值本身是动态的；`elem_type` 对应的类型对象作为模板化静态信息参与实例化，`memory_space` 继续作为静态类型信息存在

### specialization key 语义

`specialization key` 应按模板参数的规范化结果比较，而不是按 Python 对象身份比较。

统一口径：

- 模板参数 = 类型对象参数 + `pto.const(...)` 静态值参数
- `specialization key` = 模板参数的规范化结果

最小约束：

- key 包含两部分：
  - 类型对象参数
  - 被 `pto.const(...)` 标记的静态字段
- 复合值按字段递归规范化后参与 key
- list / tuple 这类顺序容器按元素值和顺序比较
- 两次 `.jit(...)` 只要类型对象参数相同，且 `pto.const(...)` 静态值参数规范化结果相同，就应命中同一个 specialization
- `memory_space` 仍是 DSL 函数签名的静态类型前提，不属于这版模板参数

例如：

```python
src0 = Tile(
    ub_ptr=pto.ptr(pto.f32, "ub"),
    shape=pto.const([32, 32]),
)

src1 = Tile(
    ub_ptr=pto.ptr(pto.f32, "ub"),
    shape=pto.const([32, 32]),
)
```

这里 `src0` 和 `src1` 绑定出的类型对象参数和 `shape` 这个 `pto.const(...)` 静态值参数完全相同，因此应落到同一个 specialization key。

相对地：

```python
src2 = Tile(
    ub_ptr=pto.ptr(pto.f32, "ub"),
    shape=pto.const([64, 32]),
)
```

这里 `shape` 已变化，因此 `src2` 应产生不同的 specialization key，并触发新的实例化结果。

再例如：

```python
compiled_f32 = abs_kernel.jit(src=src_tile_f32, dst=dst_tile_f32)
compiled_f16 = abs_kernel.jit(src=src_tile_f16, dst=dst_tile_f16)
```

即使 `shape` 完全相同，只要 `src_tile_f32.ub_ptr` 和 `src_tile_f16.ub_ptr` 的类型对象参数不同，二者也应落到不同的 specialization key。这样同一份 kernel body 就可以分别实例化出 `f32` 和 `f16` 版本，降低库开发者重复实现的成本。

## caller 与 JIT 的职责划分

示意：

```python
@pto.struct
class Tile:
    ub_ptr: pto.ptr
    shape: pto.const

@pto.vkernel
def abs_kernel(src: Tile, dst: Tile):
    ...

src_tile = Tile(
    ub_ptr=pto.ptr(pto.f32, "ub"),
    shape=pto.const([32, 32]),
)

dst_tile = Tile(
    ub_ptr=pto.ptr(pto.f32, "ub"),
    shape=pto.const([32, 32]),
)

compiled = abs_kernel.jit(src=src_tile, dst=dst_tile)
```

职责划分：

- caller:
  - 构造 `@pto.struct` 参数对象实例
  - 决定哪些字段通过 `pto.const(...)` 标记为静态值
- `kernel.jit(...)`:
  - 读取 tile 实例
  - 识别 `pto.const(...)`
  - 完成静态/动态拆分
  - 生成 specialization key
  - 触发实例化
  - 其中类型对象参数和 `pto.const(...)` 静态值参数一起决定模板化结果

## kernel body 中的 `tile` 访问

首版建议让 `@pto.struct` 参数对象在 kernel body 中保持显式对象语义，而不是在进入 body 前被偷偷拆平。

也就是说，用户在 DSL 中直接写：

- `src.ub_ptr`
- `src.shape`

其中：

- `src.ub_ptr` 的类型对象参数在实例化时可变化；`memory_space` 继续由 DSL 函数签名静态确定；实例化后只有指针值本身对应动态 runtime 入参来源
- `src.shape` 若由 `pto.const(...)` 绑定，则在实例化阶段被折叠为静态元数据
- body 内仍以统一的 `tile` 字段访问语法表达，不要求用户为模板化场景额外改写函数体

这种写法可以把“用户看到的参数结构”和“JIT 绑定后的 ABI 收敛”分开：

- 表面 DSL 仍以 `tile` 为参数对象
- 例如 `Tile` 只是用户定义的一个具体参数结构名；机制本身来自 `@pto.struct`
- JIT/实例化阶段再决定哪些字段被折叠为常量、哪些字段继续进入 runtime ABI

## 编译阶段

建议拆成五个阶段：

1. Python AST 解析阶段
- 解析 `@pto.vkernel`
- 识别 `@pto.struct` 类型参数位
- 建立 kernel 形参中的参数结构占位模型

2. JIT 绑定阶段
- caller 构造具体 `@pto.struct` 参数对象
- `kernel.jit(...)` 读取 tile 实例
- 提取 `ub_ptr`、`shape`
- 根据 `pto.const(...)` 完成静态/动态拆分
- 将静态部分收集到 specialization key
- 将动态部分保留为 runtime 参数来源
- 其中 `ub_ptr` 只保留动态指针值进入 runtime ABI；类型对象参数进入模板化层，`memory_space` 继续留在静态类型层

3. DSL 实例化阶段
- 读取 `pto.const(...)` 绑定出的静态元数据
- 在 DSL 语义层完成 `shape` 常量替换与实例化
- 产出“已实例化 DSL IR”

4. runtime ABI 收敛阶段
- 将 tile 的动态指针值收敛为最终 kernel 入参
- 所有 `pto.const(...)` 绑定值都不进入 runtime ABI

5. VPTO lowering 阶段
- 将“已实例化 + 已 ABI 收敛”的 DSL IR 降为 VPTO MLIR
- 此时 MLIR pass 不再承担语言级模板特化
- 后续 pass 只做 inline、canonicalize、CSE、lowering 等常规优化

## 内部表示

建议 `@pto.struct` 参数对象在 DSL IR 内部显式存在，而不是只在 Python 注解里短暂出现。

最小内部模型：

```text
KernelSignature
  struct_params:
    - src: Tile
        ub_ptr: !pto.ptr<T, ub>
        shape: const([32, 32])
    - dst: Tile
        ub_ptr: !pto.ptr<T, ub>
        shape: const([32, 32])
  template_params:
    - T = elem_type(src.ub_ptr) = elem_type(dst.ub_ptr)
      这里 `T` 表示类型对象参数，对应 `src.ub_ptr` / `dst.ub_ptr` 的 `elem_type` 字段值
  runtime_abi:
    - src.ub_ptr
    - dst.ub_ptr
```

## 与后端 pass 的边界

明确边界：

- DSL/Python 侧负责参数实例化
- VPTO/MLIR pass 不负责执行 Python 组件做二次特化
- VPTO/MLIR pass 只消费实例化后的 kernel IR
- 若需要 library inline，则在实例化后的 VPTO IR 上完成

## 模板化 `abs` 用例

参考现有 `abs` 语义，首版模板化用例应明确为“tile 级实现”：

- caller 负责准备好 UB tile，并把 `ub_ptr/shape` 封装进 `Tile`
- 这个模板化 kernel 只描述 tile 内的计算语义，不负责 GM <-> UB 搬运
- 当 `shape=[32, 32]` 时，它表达的核心 vector 计算语义应与现有 `abs` 测例中的 vecscope 主体一致

示例：

```python
@pto.struct
class Tile:
    ub_ptr: pto.ptr
    shape: pto.const

@pto.vkernel
def abs_kernel(src: Tile, dst: Tile):
    total = src.shape[0] * src.shape[1]
    # `elem_bytes` 来自模板化类型对象参数；动态部分只有指针值本身。
    step = 256 // src.ub_ptr.elem_bytes

    with pto.strict_vecscope(
        src.ub_ptr, dst.ub_ptr, 0, total, step, total
    ) as (
        vin, vout, lb, ub, vec_step, rem0
    ):
        rem = rem0
        for lane in range(lb, ub, vec_step):
            mask, rem = pto.plt_b32(rem)
            vec = pto.vlds(vin, lane)
            out = pto.vabs(vec, mask)
            pto.vsts(out, vout, lane, mask)
```

对应的 caller 侧示意：

```python
src_tile = Tile(
    ub_ptr=pto.ptr(pto.f32, "ub"),
    shape=pto.const([32, 32]),
)

dst_tile = Tile(
    ub_ptr=pto.ptr(pto.f32, "ub"),
    shape=pto.const([32, 32]),
)

compiled = abs_kernel.jit(src=src_tile, dst=dst_tile)

# 同一份 kernel body 也可以实例化为 f16 版本。
src_tile_f16 = Tile(
    ub_ptr=pto.ptr(pto.f16, "ub"),
    shape=pto.const([32, 32]),
)

dst_tile_f16 = Tile(
    ub_ptr=pto.ptr(pto.f16, "ub"),
    shape=pto.const([32, 32]),
)

compiled_f16 = abs_kernel.jit(src=src_tile_f16, dst=dst_tile_f16)
```

对上面这个实例，实例化后的关键语义可以近似理解为：

```python
@pto.vkernel
def abs_kernel_instantiated(
    src_ub: pto.ptr(pto.f32, "ub"),
    dst_ub: pto.ptr(pto.f32, "ub"),
):
    total = 1024
    step = 64

    with pto.strict_vecscope(
        src_ub, dst_ub, 0, total, step, total
    ) as (
        vin, vout, lb, ub, vec_step, rem0
    ):
        rem: pto.i32 = rem0
        for lane in range(lb, ub, vec_step):
            mask, rem = pto.plt_b32(rem)
            vec = pto.vlds(vin, lane)
            out = pto.vabs(vec, mask)
            pto.vsts(out, vout, lane, mask)
```

这个模板化 `abs` 用例要表达的约束是：

- `shape` 信息明确挂在 `Tile` 这类 `@pto.struct` 参数对象上
- kernel body 通过 `src.shape[i]` / `src.ub_ptr` 直接访问这些字段
- 这个用例当前是 tile 级计算 kernel，输入输出都是 UB tile，而不是完整的 GM <-> UB 搬运 kernel
- `src.ub_ptr` / `dst.ub_ptr` 只有指针值本身是动态的；类型对象参数进入模板化维度，`memory_space` 属于静态指针类型对象的一部分
- vector loop 的 `step` 由 `pto.ptr` 的模板化类型对象参数在编译期推导得到；源码侧不应手写元素字节数，实例化到 `!pto.ptr<f32, ub>` 时再落成 `256 // 4 = 64`
- `total` / `step` 这类完全由静态元数据推导出的中间量，不要求用户显式写机器整数类型，应由后续 op schema 和结构化语义上下文完成定型
- 类型对象参数与 `shape` 这类 `pto.const(...)` 静态值参数一起在 `.jit(...)` 后进入 specialization key，并在实例化阶段固定
- 最终 runtime ABI 只保留动态 pointer value，不重复传递 `pto.ptr` 的元素类型和内存空间
- 当 `shape=[32, 32]` 时，同一份 `abs` kernel body 应至少能实例化出 `f32` 和 `f16` 两个版本；其中 `f32` 版本的 vecscope 核心计算语义应能对齐已有 `abs` 测例中的主计算段

## 首版范围

首版建议只支持：

- `@pto.struct`
- 参数对象中的动态指针值字段 `ub_ptr`
- 参数对象中的 `shape`
- `pto.const(...)`
- 基于类型对象参数和 `pto.const(...)` 静态值参数的单层实现实例化

首版暂不支持：

- 多版本自动搜索
- 运行时 dispatch
- 通用 `argstruct`
- 通用 `static(...)` / `static_type()`
- shape 的部分动态化
- 复杂模板缓存

## 分步计划

Step 1. `@pto.struct` 数据模型
- 增加 `@pto.struct class Tile: ...`
- 增加 DSL IR 数据结构表示参数结构
- 验证 caller 可以独立构造参数结构对象

Step 2. JIT 绑定与拆分
- 给 `kernel.jit(...)` 增加 `@pto.struct` 参数绑定逻辑
- 在绑定阶段提取 `ub_ptr`、`shape`
- 支持 `pto.const(...)` 驱动的静态/动态拆分
- 明确 `ub_ptr` 的类型对象参数进入模板化绑定，`memory_space` 继续来自静态指针类型对象，只有指针值进入动态绑定

Step 3. 通用静态绑定标记
- 定义 `pto.const(...)` 的绑定语义
- 约束被 `pto.const(...)` 标记的值不会进入最终函数 ABI
- 约束这些值在 lowering 时按字面量/常量处理

Step 4. DSL 实例化上下文
- 在 DSL builder 中加入 tile specialization context
- 支持 `pto.const(...)` 绑定值进入 DSL IR 实例化
- 支持用 tile 元数据消解受限的常量表达式

Step 5. runtime ABI 收敛
- 将参数结构中的动态指针值收敛为最终 kernel 入参
- `pto.const(...)` 绑定值只保留在实例化结果里，不进入 runtime ABI

Step 6. 实例化后导出 VPTO
- 以“实例化后的单个 kernel”生成 VPTO MLIR
- 保持现有 `vecscope` / `strict_vecscope` / `scf.for` lowering 逻辑不变

Step 7. inline 与后续优化验证
- 构造一个“高层 tile 调用 -> 实例化实现 -> inline”样例
- 验证 inline 后 IR 还能继续跑现有 VPTO 优化链

## 验证样例建议

建议补两类样例：

1. 固定 shape 的实例化样例
- 用 `shape=pto.const([32, 32])`
- 检查生成的 VPTO 与手写 `abs` 样例一致

2. 多实例复用样例
- 同一份 DSL 定义分别实例化为不同 `shape` / 类型对象参数
- 检查只有参数结构元数据相关 IR 不同，runtime ABI 仍只保留动态指针值输入
