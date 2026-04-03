# VPTO `@pto.vkernel` Python DSL 草案

## 状态

本文档是 VPTO Python DSL 的收敛后草案，用来回答两类问题：

- 对外的 Python authoring surface 应长什么样
- 该 surface 应如何与现有 VPTO MLIR surface、typed-mask verifier 和 `vecscope` 契约对齐

本文档不是模板/JIT 设计文档。模板化与静态绑定相关内容继续放在
[`VPTO_VKERNEL_TEMPLATE_BINDING_PLAN.md`](./VPTO_VKERNEL_TEMPLATE_BINDING_PLAN.md)。

## 目标

- 面向库开发者和外部用户，提供一套可以直接 author VPTO IR 的 Python DSL
- 以 `@pto.vkernel` 作为 kernel authoring 入口
- 保留少量 Pythonic 语法糖，但绝大多数 op/type 命名与 VPTO surface 对齐
- 同时支持 `ptr` 与 `memref` 两种 authoring form
- 明确 DSL surface、语义模型、VPTO lowering contract 的分层边界

## 非目标

当前草案不追求：

- 支持任意 Python 语法
- 把 Python 运行时对象语义直接当作 IR 语义
- 在本文件中一并定义模板/JIT 绑定规范
- 为每个历史实验别名保留同等级 canonical 地位

## 规范收敛依据

本草案主要参考以下材料：

- [`docs/vpto-spec.md`](../docs/vpto-spec.md)
- [`docs/vpto-verify.md`](../docs/vpto-verify.md)
- `docs/isa/*.md` 细分 ISA 文档
- [`include/PTO/IR/VPTOOps.td`](../include/PTO/IR/VPTOOps.td)
- [`python/pto/dialects/pto.py`](../python/pto/dialects/pto.py) 中现有实验实现

当 `docs/vpto-spec.md` 的 merged draft 与细 ISA 文档、真实 op 定义或 verifier
契约冲突时，本草案按以下优先级收敛：

1. 细 ISA 文档
2. `include/PTO/IR/VPTOOps.td`
3. `docs/vpto-verify.md`
4. `docs/vpto-spec.md`

## 分层模型

本草案将 DSL 分成三层，避免把 MLIR 细节和 Python surface 混在一起。

### 1. Python Surface

用户可见的 DSL surface，包括：

- `@pto.vkernel`
- `pto.ptr(...)`
- `pto.memref(...)`
- `pto.vreg(...)`
- `pto.mask_b8` / `pto.mask_b16` / `pto.mask_b32`
- `pto.align`
- `with pto.vecscope():`
- `with pto.strict_vecscope(...) as (...)`
- `for range(lb, ub, step)`
- `if`
- `pto.*` canonical op family

### 2. DSL 语义模型

负责解释 Python source 中的：

- 值模型
- 绑定环境
- literal typing
- branch merge
- loop-carried state
- capture 规则
- diagnostics

### 3. VPTO Lowering Contract

负责把 DSL 语义模型中的结构映射到 MLIR：

- kernel -> `func.func`
- `vecscope` -> `pto.vecscope`
- `strict_vecscope` -> `pto.strict_vecscope`
- counted loop -> `scf.for`
- conditional -> `scf.if`
- DSL op -> `pto.*` / `arith.*` / `scf.*`

该层只说明 lowering 目标和接口契约，不在草案中展开 pass 级实现细节。

## Canonical Naming Policy

### 总原则

- 对外 canonical 名称优先与 VPTO 规范口径对齐
- 但遇到 merged draft 与真实 op 定义冲突时，以真实 op 名为准
- 旧实验别名只作为 compatibility note，不再作为一等 surface

### 已明确收敛的冲突点

#### `pto.pipe_barrier`

Python DSL 的 canonical 名称为：

```python
pto.pipe_barrier("PIPE_ALL")
```

当前仓库实验实现仍存在 `pto.barrier(...)` 路径。该名字在后续实现中可以保留为兼容别名，但不是 canonical surface。

#### Predicate materialization / movement family

对外 canonical 名称按真实 op 与细 ISA 文档收敛为：

- `pto.pset_b8` / `pto.pset_b16` / `pto.pset_b32`
- `pto.pge_b8` / `pto.pge_b16` / `pto.pge_b32`
- `pto.plt_b8` / `pto.plt_b16` / `pto.plt_b32`
- `pto.ppack`
- `pto.punpack`
- `pto.pnot`
- `pto.psel`
- `pto.pdintlv_b8`
- `pto.pintlv_b16`
- `pto.psts`
- `pto.pstu`

因此，`docs/vpto-spec.md` merged draft 中出现的这些写法不作为 canonical Python 名称：

- `vpset_*`
- `vpge_*`
- `vppack`
- `vpunpack`
- `vpnot`
- `vpsel`
- `vpdintlv_b8`
- `vpintlv_b16`
- `vpsts`
- `vpstu`

## 用户入口

推荐入口：

```python
import ... as pto

@pto.vkernel(target="a5", name="abs_kernel")
def abs_kernel(src: pto.ptr(pto.f32, "gm"),
               dst: pto.ptr(pto.f32, "gm")):
    ...
```

`@pto.vkernel` 返回的应是 kernel descriptor，而不是普通 Python 可执行函数。
descriptor 可提供：

- `mlir_text()`
- `mlir_module()`
- `verify()`
- `dump()`
- `emit(path)`

导入路径不属于 DSL 语义契约。本文统一用 `pto` 作为命名空间别名。

## 类型系统

### 标量类型

DSL 暴露以下机器标量类型对象：

- `pto.i1`
- `pto.i8`
- `pto.i16`
- `pto.i32`
- `pto.i64`
- `pto.f16`
- `pto.bf16`
- `pto.f32`

Python 字面量仍可用作 surface syntax：

- `bool`
- `int`
- `float`

但它们不是 DSL 的最终类型系统本体。

### Typed Mask

DSL 不再使用模糊的 `mask` surface。
canonical typed-mask surface 为：

- `pto.mask_b8`
- `pto.mask_b16`
- `pto.mask_b32`

对应 VPTO 类型：

- `pto.mask_b8` -> `!pto.mask<b8>`
- `pto.mask_b16` -> `!pto.mask<b16>`
- `pto.mask_b32` -> `!pto.mask<b32>`

规则：

- `pset_b32` / `pge_b32` / `plt_b32` 只能生成 `mask_b32`
- `pset_b16` / `pge_b16` / `plt_b16` 只能生成 `mask_b16`
- `pset_b8` / `pge_b8` / `plt_b8` 只能生成 `mask_b8`
- 消费 mask 的向量 family 必须与向量元素家族匹配

### 向量类型

`pto.vreg(lanes, elem_type)` 表示 DSL surface 上的 VPTO 向量类型。

约束：

- `vreg` 总宽度必须固定为 256B
- 它不是任意宽度向量容器
- `lanes * bitwidth(elem_type) = 2048`

例如：

```python
pto.vreg(64, pto.f32)
pto.vreg(128, pto.f16)
```

### 指针类型

指针 surface：

```python
pto.ptr(elem_type, space)
```

第一阶段主要使用：

- `space == "gm"`
- `space == "ub"`

### MemRef 类型

为了支持 VPTO `buf_like` authoring form，DSL 引入 canonical memref surface：

```python
pto.memref(shape, elem_type, space)
```

约束：

- `shape` 是 DSL 类型层的 shape 说明，不是运行时值对象
- `space` 表示逻辑内存域，例如 `"gm"` / `"ub"`
- lowering 时允许把该 surface 映射为目标 MLIR memref address-space form

该 surface 的目标是 authoring `buf_like` family，不是替代 `pto.ptr(...)` 的所有用途。

### `pto.align`

`pto.align` 映射到 `!pto.align`，仅用于 align carrier family。

### Index 原则

用户层不直接把 `index` 暴露成需要显式注解的日常类型。

收敛规则：

- `range(lb, ub, step)` 的边界值在 loop lowering 位置可按 index-like 语义解释
- `strict_vecscope` capture 中用于 loop bound / offset 的值允许被上下文定型为 index-like
- 普通机器整数仍通过 `pto.i32` / `pto.i64` 表达
- 任何需要跨结构流动并进入 IR 数据流边界的机器整数，应显式写成 `pto.i32(...)` / `pto.i64(...)` 或使用注解

## 值模型与 literal typing

DSL 中的值不是 Python 运行时值，而是符号值。

至少要区分：

- 常量值
- op 结果值
- block argument

字面量规则：

- `bool` 默认收敛到 `i1`
- `float` 默认收敛到 `f32`
- `int` 允许在表达式级上下文中被定型
- 一旦进入 IR 数据流边界，必须是显式机器类型，不能靠跨结构全局猜测
- 标量类型对象可作为 literal constructor 使用，例如 `pto.i32(1024)`

推荐写法：

```python
remaining: pto.i32 = 1024
remaining = pto.i32(1024)
```

不推荐把裸 `int` 直接带入结构化 loop-carried / branch-merge 边界。

## 支持的 Python 子集

### 支持

- 顶层 `def`
- 参数类型注解
- 赋值
- 带注解赋值
- tuple unpack
- 表达式语句
- 空 `return`
- `with pto.vecscope():`
- `with pto.strict_vecscope(...) as (...)`
- `for i in range(lb, ub, step)`
- `if`
- Python 字面量常量

### 暂不支持

- `while`
- `break` / `continue`
- 推导式
- `try` / `except`
- `lambda`
- `class`
- 任意闭包自由捕获

## 统一绑定环境模型

`if`、`for`、`vecscope`、`strict_vecscope` 都在统一绑定环境模型下分析，但它们向外暴露结果的方式并不相同。

### `if`

- 分支内部可更新已有绑定或创建新绑定
- 两个分支都更新的外层绑定，按统一 merge 机制向外暴露
- 仅在分支内部创建的局部绑定不会泄漏到外部

### `for`

- 通过统一迭代状态传递机制表达 loop-carried state
- 用户不直接写 `yield`
- 语义层将其收敛到 `scf.for` 的 `iter_args` / `scf.yield`

### `vecscope` / `strict_vecscope`

- 它们是 region boundary，而不是 outward merge result constructor
- body 内允许存在 loop-carried state
- scope 本身在 v1 不向外返回结构结果，lowering 结果为 `()`
- 因此，统一绑定环境模型在 scope 内仍适用，但 scope 结束时不会额外产生 outward merge value

## `pto.vecscope` 语义

canonical surface：

```python
with pto.vecscope():
    ...
```

语义：

- body 可隐式引用外层 DSL 绑定
- lowering 到 `pto.vecscope`
- scope 自身不向外返回结构化结果

## `pto.strict_vecscope` 语义

canonical surface：

```python
with pto.strict_vecscope(a, b, c) as (a_, b_, c_):
    ...
```

语义：

- `a, b, c` 是显式 capture
- `a_, b_, c_` 是 block arguments
- body 不允许隐式捕获外层值
- body 只能访问：
  - `as (...)` 引入的 block arg 名
  - body 内局部新定义名
  - 字面量
- lowering 到 `pto.strict_vecscope`

典型错误信息应接近：

```text
strict_vecscope body cannot capture outer value 'ub_in' implicitly; pass it in the capture list and use the bound region argument instead
```

## Address-Form Policy

DSL v1 同时支持 `ptr` 与 `memref` 两种 authoring form，但不是所有 family 都放宽为二者都可用。

### `ptr` / `memref` 都可 author 的 family

主要是 VPTO 中接受 `buf_like` 的 stateless / predicate family，例如：

- `vlds`
- `vldx2`
- `vsld`
- `vsts`
- `vstx2`
- `psts`
- `vsst`
- `vsta`
- `vstar`

### 仍保持 pointer-only 或 buffer-only 语义的 family

这类 family 继续遵守底层 VPTO 契约，不因为 DSL 引入 `pto.memref(...)` 而放宽：

- pointer construction family，如 `castptr` / `addptr`
- stateful store family，如 `pstu` / `vstu` / `vstus` / `vstur`
- gather / scatter / sort 这类当前契约仍要求 pointer-like buffer 的 family
- copy programming / copy transfer 中需要遵守现有 `PTO_BufferType` 约束的 family

## Family Mapping 策略

正式 DSL spec 不再用 ad-hoc op 清单描述 surface，而是按 VPTO family 分组。

### Pointer Construction

- `castptr`
- `addptr`

### 1. Sync And Buffer Control

- `set_flag`
- `wait_flag`
- `pipe_barrier`
- `get_buf`
- `rls_buf`

### 2. Copy Programming

- `set_loop2_stride_outtoub`
- `set_loop1_stride_outtoub`
- `set_loop_size_outtoub`
- `set_loop2_stride_ubtoout`
- `set_loop1_stride_ubtoout`
- `set_loop_size_ubtoout`

### 3. Copy Transfers

- `copy_gm_to_ubuf`
- `copy_ubuf_to_ubuf`
- `copy_ubuf_to_gm`

### 4. Vector / Predicate / Align Loads

- `vlds`
- `vldas`
- `vldus`
- `vplds`
- `vpld`
- `vpldi`
- `vldx2`
- `vgather2`
- `vgatherb`
- `vgather2_bc`
- `vsld`
- `vsldb`

### 5. Materialization And Predicate Construction

- `vbr`
- `vdup`
- `pset_b*`
- `pge_b*`
- `plt_b*`
- `ppack`
- `punpack`

### 6. Unary Vector Ops

- `vabs`
- `vexp`
- `vln`
- `vsqrt`
- `vrec`
- `vrelu`
- `vnot`
- `vcadd`
- `vcmax`
- `vcmin`
- `vbcnt`
- `vcls`

### 7. Binary Vector Ops

- `vadd`
- `vsub`
- `vmul`
- `vdiv`
- `vmax`
- `vmin`
- `vand`
- `vor`
- `vxor`
- `vshl`
- `vshr`

### 8. Vec-Scalar Ops

- `vmuls`
- `vadds`
- `vmaxs`
- `vmins`
- `vlrelu`
- `vshls`
- `vshrs`

### 9. Carry / Compare / Select

- `vaddc`
- `vsubc`
- `vaddcs`
- `vsubcs`
- `vsel`
- `vselr`
- `vselrv2`
- `vcmp`
- `vcmps`
- `pnot`
- `psel`

### 10. Pairing And Interleave

- `pdintlv_b8`
- `pintlv_b16`
- `vintlv`
- `vdintlv`
- `vintlvv2`
- `vdintlvv2`

### 11. Conversion / Index / Sort

- `vtrc`
- `vcvt`
- `vci`
- `vbitsort`
- `vmrgsort4`

### 12. Extended Arithmetic

- `vmull`
- `vmula`

### 13. Stateless Stores

- `vsts`
- `vscatter`
- `vsts_pred`
- `psts`
- `pst`
- `psti`
- `vsst`
- `vstx2`
- `vsstb`
- `vsta`
- `vstas`
- `vstar`

### 14. Stateful Store Ops

- `pstu`
- `vstu`
- `vstus`
- `vstur`

## 示例

### 1. `ptr` 版 `abs` kernel

```python
@pto.vkernel(target="a5", name="abs_kernel")
def abs_kernel(src: pto.ptr(pto.f32, "gm"),
               dst: pto.ptr(pto.f32, "gm")):
    pto.set_loop_size_outtoub(1, 1)

    ub_in = pto.castptr(0, pto.ptr(pto.f32, "ub"))
    ub_out = pto.castptr(4096, pto.ptr(pto.f32, "ub"))
    pto.copy_gm_to_ubuf(src, ub_in, 0, 32, 128, 0, 0, False, 0, 128, 128)

    pto.set_flag("PIPE_MTE2", "PIPE_V", "EVENT_ID0")
    pto.wait_flag("PIPE_MTE2", "PIPE_V", "EVENT_ID0")

    with pto.strict_vecscope(ub_in, ub_out, 0, 1024, 64, pto.i32(1024)) as (
        vin, vout, lb, ub, step, rem0
    ):
        rem: pto.i32 = rem0
        for lane in range(lb, ub, step):
            mask, rem = pto.plt_b32(rem)
            vec = pto.vlds(vin, lane)
            out = pto.vabs(vec, mask)
            pto.vsts(out, vout, lane, mask)

    pto.set_flag("PIPE_V", "PIPE_MTE3", "EVENT_ID0")
    pto.wait_flag("PIPE_V", "PIPE_MTE3", "EVENT_ID0")
    pto.copy_ubuf_to_gm(ub_out, dst, 0, 32, 128, 0, 128, 128)
    pto.pipe_barrier("PIPE_ALL")
```

### 2. `memref` 版 stateless load/store

```python
@pto.vkernel(name="copy_line")
def copy_line(src: pto.memref(256, pto.f32, "ub"),
              dst: pto.memref(256, pto.f32, "ub")):
    with pto.vecscope():
        all_mask: pto.mask_b32 = pto.pset_b32("PAT_ALL")
        for offset in range(0, 256, 64):
            vec = pto.vlds(src, offset)
            pto.vsts(vec, dst, offset, all_mask)
```

### 3. `if` merge 结果

```python
flag: pto.i1 = some_flag
step: pto.i32 = 64

if flag:
    step = pto.i32(64)
else:
    step = pto.i32(128)

# if 结束后的 step 等价于 scf.if 的 merge result
```

### 4. Typed-mask 正反例

合法：

```python
all_mask: pto.mask_b32 = pto.pset_b32("PAT_ALL")
vec = pto.vlds(src, offset)
out = pto.vabs(vec, all_mask)
```

非法：

```python
bad_mask: pto.mask_b16 = pto.pset_b16("PAT_ALL")
out = pto.vabs(vec_f32, bad_mask)  # f32 family 不能消费 mask_b16
```

## 当前实验实现与草案的主要差异

当前 [`python/pto/dialects/pto.py`](../python/pto/dialects/pto.py) 中的实验实现尚未追上本草案，至少包括：

- 仍使用未分型 `mask`
- 还没有 `pto.memref(...)`
- 仍暴露 `pto.barrier(...)`，没有以 `pto.pipe_barrier(...)` 为 canonical surface
- 当前仅覆盖很小的 op 子集
- 目前 `vecscope` / `strict_vecscope` 的实现可作为概念验证，但不等同于最终规范

这些差异应在正式 spec 中作为 compatibility note 记录，而不是反向约束语言规范。

## Diagnostics

DSL 应尽量报源位置相关错误，例如：

- 不支持的 Python 语法
- 未定义符号
- 类型无法定型
- typed-mask 粒度不匹配
- `strict_vecscope` 非法隐式捕获
- region 参数数量不匹配
- 非空 `return`
- vecscope placement 相关错误

## 当前结论

- DSL canonical surface 应显式引入 typed-mask，而不是延续旧 `!pto.mask`
- `vecscope` / `strict_vecscope` 是 region boundary，不是 outward merge constructor
- DSL v1 同时支持 `ptr` 与 `memref` authoring form，但 family 合法性仍必须服从 VPTO 既有契约
- 合并稿中存在的命名和签名冲突，必须在正式 spec 中集中写清，不能让实现层隐式兜底
