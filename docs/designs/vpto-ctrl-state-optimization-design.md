# VPTO `get_ctrl/set_ctrl` 优化设计

## 1. 问题背景与优化目标

### 1.1 问题背景

[PTOAS issue #1279](https://github.com/hw-native-sys/PTOAS/issues/1279) 显示，PTO 与
Ascend 后端使用相同的前端调度、算法和主要 CUBE/MTE 指令参数，但在 12 个 GEMM
用例上慢 1% 到 8.2%。这些用例的 Scalar 利用率均有所上升；在 MAD-bound 用例中，
MAD 利用率通常下降 3 到 5 个百分点。性能差距主要来自额外的标量指令。

主要额外开销来自 MAD semantic-to-raw lowering。对普通非 HiF8、TF32 disabled、
未指定 sat 且没有 `n_dir` 的 MAD，当前实现按固定模板为每个 `mad_raw` 生成：

```text
%saved = pto.get_ctrl
%c1 = pto.sbitset0 %saved, 45
%c2 = pto.sbitset0 %c1, 46
%c3 = pto.sbitset0 %c2, 47
%active = pto.sbitset0 %c3, 51
pto.set_ctrl %active
pto.mad_raw ...
pto.set_ctrl %saved
```

每个 MAD 因此增加 1 次读取、4 次位更新和 2 次写入，并形成
`get_ctrl -> bit update -> set_ctrl -> mad_raw` 的标量依赖链。连续 MAD 即使使用
相同的 CTRL 配置，也会重复保存、配置和恢复。

`VPTOExpandWrapperOpsPass` 在 semantic-to-raw 阶段就把 MAD 的临时 CTRL 要求物化为
局部 `get_ctrl/set_ctrl` 序列。物化后，临时状态的作用域和 requirement 不再可见，
后续 pass 无法可靠地区分配置写、恢复写和独立 CTRL 访问。

### 1.2 优化目标

本文只优化 MAD lowering 产生的 CTRL `get_ctrl/set_ctrl` 往返，目标是：

1. 显式表示 MAD 的临时 CTRL requirement，而不是在 semantic lowering 阶段立即
   生成保存、配置和恢复序列；
2. 在保证 CTRL 可观察语义不变的前提下，让连续兼容的 MAD 共享 CTRL 读取和写入，
   并将恢复延迟到真正的观察点或状态边界；
3. 在 VPTO emission 前消除内部状态表示，只物化语义必要的
   `get_ctrl/set_ctrl`；
4. 不依赖特定 dtype、GEMM 形状或“入口 CTRL 为零”等场景假设；
5. 用通用接口描述状态访问和临时 requirement，便于后续 CTRL 使用者接入。

首版不优化 AccStore、MASK、atomic 或其他特殊状态，也不改变它们当前的 lowering。
这些路径已有的独立 `get_ctrl/set_ctrl` 保留为优化边界。方案的通用性来自可扩展的
IR 接口和状态算法，不表示首版要同时接入这些使用者。

## 2. 当前 PTOAS 实现

### 2.1 MAD semantic-to-raw lowering

`lib/PTO/Transforms/VPTOExpandWrapperOps.cpp` 中，`lowerMadSemanticOp()` 在创建
`mad_raw` 前后直接生成 `GetCtrlOp` 和 `SetCtrlOp`。CTRL 位由 semantic op 的属性
和 operand type 推导：

| 语义 | CTRL bit |
|---|---|
| HiF8 | bit 45 |
| TF32 enable/round | bits 46-47 |
| 显式 `sat/nosat` | bit 48 |
| `n_dir` | bit 51 |

未被本次 MAD 管理的 bit 必须继承进入 MAD 前的 CTRL，不能用完整常量覆盖。

实现先用 `get_ctrl` 保存完整的 entry CTRL，再由 `buildMadSemanticCtrl()` 逐 bit
构造 active CTRL；`mad_raw` 执行后立即恢复 entry CTRL：

```c++
Value ctrlSaved = rewriter.create<pto::GetCtrlOp>(loc).getResult();
Value ctrlForOp = buildMadSemanticCtrl(loc, ctrlSaved, isHif8, tf32Mode,
                                       satMode, op.getNDir(), rewriter);
rewriter.create<pto::SetCtrlOp>(loc, ctrlForOp);
emitMadRawOp(...);
rewriter.create<pto::SetCtrlOp>(loc, ctrlSaved);
```

### 2.2 当前 IR 形态

`VPTOExpandWrapperOpsPass` 输入是 semantic MAD：

```text
pto.mad ...
pto.mad ...
```

pass 独立展开每个 semantic MAD。省略与 CTRL 无关的 `xt` 打包后，核心 IR 如下：

```text
%entry0 = pto.get_ctrl
%active0 = update_bits(%entry0, C0, R0)
pto.set_ctrl %active0
pto.mad_raw ...
pto.set_ctrl %entry0

%entry1 = pto.get_ctrl
%active1 = update_bits(%entry1, C1, R1)
pto.set_ctrl %active1
pto.mad_raw ...
pto.set_ctrl %entry1
```

`update_bits` 代表若干 `pto.sbitset0/pto.sbitset1`，不是现有 op。
即使 `(C0,R0) == (C1,R1)`，两个 MAD 之间仍然存在 restore、再次读取和再次配置。
后续 CSE 无法把这段序列当作普通纯 SSA 冗余删除，因为 `get_ctrl/set_ctrl` 访问的是
有顺序语义的隐式硬件状态。

### 2.3 现有 pipeline

VPTO unified emission pipeline 在 `tools/ptoas/ptoas.cpp` 的
`prepareVPTOForEmission()` 中组装，相关顺序如下：

```text
VPTOExpandWrapperOpsPass
VPTOInferVPTOVecScopePass
VPTOSoftPostUpdatePass (optional)
LICM / loop transforms / canonicalizer / CSE
PTOExpandSoftLibPass
PTOInlineLibCallPass
canonicalizer / CSE
VPTOSchedulerPass (optional)
VPTOCombineReductionsPass / CSE
PTOValidateVPTOEmissionIRPass
```

LLVM emission 使用另一条 pipeline。`LowerVPTOOpsPass` 中已有：

```text
pto.get_ctrl  -> llvm.hivm.GET.CTRL
pto.set_ctrl  -> llvm.hivm.SET.CTRL
```

当前没有分析 CTRL 状态、合并上述往返的 pass。`LowerVPTOOpsPass` 只做逐 op
intrinsic lowering，不判断 `get_ctrl/set_ctrl` 是否冗余。

## 3. 优化方案设计

### 3.1 核心思路

核心思路是把 CTRL requirement 的推导与硬件 CTRL 读写的生成分开。

`VPTOExpandWrapperOpsPass` 仍完成 MAD semantic-to-raw lowering，但不立即生成
`get_ctrl/set_ctrl`。它根据 MAD 的类型和属性计算 `(C,R)`，再用
`CtrlStateGuardOp` 包裹 raw MAD，把临时 CTRL 要求保留为结构化 IR：

```text
进入 guard 时的逻辑 CTRL = E
guard 管理的 bit 集合 = C
这些 bit 中目标值为 1 的集合 = R
consumer 执行时需要的 active CTRL = (E & ~C) | R
guard 退出后的逻辑 CTRL 仍然是 E
```

`CtrlStateGuardOp` 只描述语义，不保存 CTRL，也不规定何处配置和恢复硬件状态。
这些决策统一交给 `VPTOOptimizeCtrlStatePass`。

`VPTOOptimizeCtrlStatePass` 先分析 IR，不立即重写。pass 跟踪每个程序点的 logical
state 和 physical state，识别显式 CTRL 访问、未知调用等状态边界，并计算每个 guard
需要的完整 active state：

```text
active(E,C,R) = (E & ~C) | R
```

优化判断比较的是完整 active state，而不只是 `(C,R)` 是否相同。相同 `(C,R)` 在
logical entry 相同时通常可以共享配置；不同 `(C,R)` 如果也能证明完整 active state
相同，同样可以省略重复写入。无法证明等价时，pass 保守地保留状态切换。

分析完成后，pass 根据物化计划生成必要的 `get_ctrl`、位更新和 `set_ctrl`，展开
guard body，并删除全部 `CtrlStateGuardOp`。因此，优化器不再匹配下面这种固定序列：

```text
get_ctrl + N x sbitset + set_ctrl + mad_raw + set_ctrl
```

整体流程如下：

```text
保留结构化 CTRL 语义 -> 分析 logical/physical 状态 -> 物化必要的硬件 CTRL 读写
```

优化不依赖 bit 更新的数量、顺序或 lowering helper 的具体实现。

### 3.2 Pipeline 与 pass 职责

设计后的相关 pipeline 为：

```text
semantic MAD
    |
    v
VPTOExpandWrapperOpsPass                   // 现有 pass，修改行为
    |  生成 CtrlStateGuardOp + mad_raw
    v
VPTO 中间优化与 lowering
    |  vec scope / loop transforms / softlib / scheduler / combine / CSE
    v
VPTOOptimizeCtrlStatePass                  // 唯一新增 pass
    |  状态分析、guard 优化、get/set 物化、guard 消除
    v
PTOValidateVPTOEmissionIRPass              // 禁止内部 guard 残留
    |
    v
LowerVPTOOpsPass                           // 现有 pass，行为不变
    |  get_ctrl/set_ctrl -> LLVM intrinsic
    v
llvm.hivm.GET.CTRL / llvm.hivm.SET.CTRL
```

方案只新增 `VPTOOptimizeCtrlStatePass`。该 pass 放在可能产生或移动 CTRL 使用者的
VPTO 变换之后、emission validation 之前。`LowerVPTOOpsPass` 继续负责逐 op lowering。

### 3.3 Pipeline 中的 IR 形态

#### `VPTOExpandWrapperOpsPass` 之前

输入仍是用户语义明确的 MAD op：

```text
pto.mad ... attributes {tf32_mode, sat_mode, n_dir, ...}
pto.mad ... attributes {tf32_mode, sat_mode, n_dir, ...}
```

#### `VPTOExpandWrapperOpsPass` 之后

pass 完成 raw op 选择和 `xt` 打包，并把 CTRL 临时需求表示为 guard：

```text
pto.ctrl_state_guard controlled_bits(C0) required_bits(R0) {
  pto.mad_raw ...
}

pto.ctrl_state_guard controlled_bits(C1) required_bits(R1) {
  pto.mad_raw ...
}
```

此时不生成 `get_ctrl`、位更新或 `set_ctrl`。`C/R` 由 semantic MAD 的类型和属性
静态计算。guard 记录 active CTRL 要求，以及退出后逻辑状态恢复的语义。中间 pass
必须保留 guard 的作用域和副作用。

#### `VPTOOptimizeCtrlStatePass` 之后

对于三个 requirement 相同且中间没有状态边界的 guard：

```text
guard C,R { mad_raw_0 }
guard C,R { mad_raw_1 }
guard C,R { mad_raw_2 }
```

典型物化结果为：

```text
%entry = pto.get_ctrl
%active = update_bits(%entry, C, R)
pto.set_ctrl %active
pto.mad_raw_0
pto.mad_raw_1
pto.mad_raw_2
pto.set_ctrl %entry       // 仅在后继观察点或边界确实需要时生成
```

后继 guard 使用另一组 `(C2,R2)` 时，可直接从 logical entry 构造 `active2`，不必
先恢复 entry。若 physical CTRL 已等于目标 active CTRL，则省略 `set_ctrl`。pass
结束时必须消除所有 `CtrlStateGuardOp`，`update_bits` 只展开为必要的 `sbitset*` 或
等价位运算。

### 3.4 设计边界与通用性

首版只优化 MAD lowering 显式生成的 CTRL guard。其他 lowering 已有的独立
`get_ctrl/set_ctrl` 不在本次重写范围内，并作为状态边界处理。显式 `get_ctrl`、独立
`set_ctrl`、未知 `func.call`、inline asm 或没有状态摘要的 backend 操作，都可能观察
或改变 CTRL；无法证明安全时，必须保守物化。

算法通过接口获取状态资源、访问类型和临时 requirement，不依赖 MAD op 名称。其他
CTRL 使用者若具有相同的临时覆盖语义，可在后续接入；AccStore、MASK 和 atomic
不在本次范围内。

## 4. 新增内部 IR 与接口

新增一个内部 op、两个 op interface 和两个枚举，仅供 authoring-to-emission 阶段
使用。最终 VPTO emission IR 不得保留这些内部表示。

两个接口分别描述单点状态访问和临时状态作用域：

| 接口 | 描述的对象 | 时间语义 | pass 中的用途 |
|---|---|---|---|
| `StateAccessOpInterface` | 单个状态访问/使用点 | 在该操作点 query、write、consume 或 clobber | 构建状态转移和优化边界 |
| `StateGuardOpInterface` | 包含 consumer 的临时状态作用域 | 进入时覆盖部分状态，退出后逻辑状态恢复 | 合并相邻 requirement、延迟恢复、避免重复 get/set |

`StateAccessOpInterface` 区分查询、写入和消费，`StateGuardOpInterface` 标记临时
配置的范围和目标值。pass 据此分析状态，不再猜测 op 名称或
`get_ctrl/sbitset*/set_ctrl` 的 SSA 形状。

### 4.1 `StateAccessOpInterface`

在 `include/PTO/IR/VPTOInterfaces.td` 中定义状态访问接口。首版只处理 `Ctrl`
resource，但接口不依赖 MAD：

```tablegen
def PTO_StateAccessOpInterface : OpInterface<"StateAccessOpInterface"> {
  let cppNamespace = "::mlir::pto";
  let methods = [
    InterfaceMethod<"Return the accessed state resource.",
      "StateResource", "getStateResource">,
    InterfaceMethod<"Return the access kind.",
      "StateAccessKind", "getStateAccessKind">,
    InterfaceMethod<"Return the SSA value read from or written to the state.",
      "::mlir::Value", "getAccessedStateValue">
  ];
}
```

#### 为什么需要这个接口

`get_ctrl/set_ctrl` 显式访问 CTRL，raw MAD 没有 CTRL operand，却会隐式消费当前
CTRL。统一接口让 pass 可以按访问语义处理操作，而不是按 op class 分派；后续使用者
也能接入同一套状态分析。

| Method | 语义 | pass 如何使用 |
|---|---|---|
| `getStateResource()` | 当前操作访问哪一个隐式状态资源 | 只在相同 resource 上建立先后依赖；当前仅处理 `Ctrl` |
| `getStateAccessKind()` | 访问是 query、write、consume 还是 clobber | 选择状态转移规则 |
| `getAccessedStateValue()` | query 产生的完整状态值，或 write 写入的完整状态值 | 更新 logical/physical SSA base；consume/clobber 返回空值 |

`getAccessedStateValue()` 只返回完整寄存器值。部分 bit requirement 由
`StateGuardOpInterface` 描述，不通过这个 accessor 传递。

对应的 C++ 类型：

```c++
enum class StateResource : uint8_t {
  Ctrl,
};

enum class StateAccessKind : uint8_t {
  Query,
  Write,
  Consume,
  Clobber,
};
```

PTOAS 没有独立的 `PTOEnums.td`。enum 由 `PTOOps.td` 包含的 `PTOAttrs.td`
生成，新增定义放在 `include/PTO/IR/PTOAttrs.td`：

```tablegen
def PTO_StateResource_Ctrl : I32EnumAttrCase<"Ctrl", 0, "ctrl">;
def PTO_StateResourceEnum : PTO_I32Enum<
    "StateResource", "PTO implicit state resource",
    [PTO_StateResource_Ctrl]>;

def PTO_StateAccessKind_Query : I32EnumAttrCase<"Query", 0, "query">;
def PTO_StateAccessKind_Write : I32EnumAttrCase<"Write", 1, "write">;
def PTO_StateAccessKind_Consume : I32EnumAttrCase<"Consume", 2, "consume">;
def PTO_StateAccessKind_Clobber : I32EnumAttrCase<"Clobber", 3, "clobber">;
def PTO_StateAccessKindEnum : PTO_I32Enum<
    "StateAccessKind", "PTO implicit state access kind",
    [PTO_StateAccessKind_Query, PTO_StateAccessKind_Write,
     PTO_StateAccessKind_Consume, PTO_StateAccessKind_Clobber]>;
```

强类型 enum 统一 ODS interface、verifier 和 C++ pass 的取值；新增 resource 或
access kind 时，编译器也能暴露遗漏。enum 只用于生成 C++ 类型，不存为 guard
attribute。

| 访问类型 | 语义 | `getAccessedStateValue()` | 首版实现者 |
|---|---|---|---|
| `Query` | 读取完整状态并产生 SSA result | query result | `GetCtrlOp` |
| `Write` | 写入完整状态 | write operand | `SetCtrlOp` |
| `Consume` | 隐式消费当前状态，不产生状态 result | 空 `Value` | 四种 raw MAD |
| `Clobber` | 可能任意修改状态，操作后状态 unknown | 空 `Value` | 暂无 |

`PTO_GetCtrlOp` 不能添加 `Pure` trait。接口只描述顺序语义；某次读取能否转发，
由状态 pass 根据分析结果决定。

### 4.2 `StateGuardOpInterface`

该接口描述临时配置使用的资源、位要求和作用域：

```tablegen
def PTO_StateGuardOpInterface : OpInterface<"StateGuardOpInterface"> {
  let cppNamespace = "::mlir::pto";
  let methods = [
    InterfaceMethod<"Return the guarded state resource.",
      "StateResource", "getStateResource">,
    InterfaceMethod<"Return the state bits owned by this guard.",
      "uint64_t", "getControlledStateBits">,
    InterfaceMethod<"Return the required values for the controlled bits.",
      "uint64_t", "getRequiredStateBits">,
    InterfaceMethod<"Return the guarded body.",
      "::mlir::Region &", "getGuardedBody">
  ];
}
```

#### 为什么需要这个接口

`CtrlStateGuardOp` 是具体实现。优化算法通过 `StateGuardOpInterface` 读取所需信息，
因此不依赖具体 op 名称。

| Method | 语义 | pass 如何使用 |
|---|---|---|
| `getStateResource()` | guard 临时配置哪一个状态资源 | 防止不同物理资源之间错误合并 |
| `getControlledStateBits()` | guard 拥有并覆盖的 bit 集合 | 确定哪些 bit 从 requirement 取值、哪些 bit 从 entry state 继承 |
| `getRequiredStateBits()` | controlled bits 中目标值为 1 的集合 | 构造 active state，并比较两个 requirement 是否等价 |
| `getGuardedBody()` | 必须在 active state 下执行的 region | 确定配置的生效区间，并将其中 consumer 作为原子状态使用点 |

接口不返回保存值，也不规定恢复位置，这些工作由优化 pass 完成。首版只接受
`StateResource::Ctrl`，resource 参数为后续状态类型预留。

当前 `uint64_t C/R` 只适用于编译期已知、最多 64 bit 的部分覆盖。其他状态若使用
动态 SSA 值、多个寄存器或非 bit 编码，需要扩展 guard interface，不能复用现有
`controlled_bits/required_bits` 的含义。

### 4.3 `CtrlStateGuardOp`

在 `include/PTO/IR/VPTOOps.td` 中增加内部 op。`controlled_bits` 和
`required_bits` 表示 CTRL 的 bit 集合，不是硬件向量 MASK 状态：

```tablegen
def PTO_CtrlStateGuardOp : PTO_Op<"ctrl_state_guard", [
    DeclareOpInterfaceMethods<PTO_StateGuardOpInterface>,
    SingleBlock,
    NoTerminator,
    RecursiveMemoryEffects
  ]> {
  let arguments = (ins I64Attr:$controlled_bits,
                       I64Attr:$required_bits);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `controlled_bits` `(` $controlled_bits `)`
    `required_bits` `(` $required_bits `)`
    $body attr-dict
  }];
}
```

#### 为什么需要这个具体 op

`StateGuardOpInterface` 只声明方法，不能存储 region、attributes 或 verifier。
`CtrlStateGuardOp` 用结构化作用域保存临时配置，使 pass 能把 guard 和 raw MAD 作为
一个整体处理，并在 emission 前完成校验和消除。它不是硬件指令，也不对应单独的
`get_ctrl` 或 `set_ctrl`。

#### 为什么不把 requirement 直接挂到 `mad_raw`

把 `controlled_bits/required_bits` 挂到 `mad_raw` 只能描述执行点的配置，表达不了：

* 配置是临时覆盖，guard 退出后的逻辑 CTRL 必须回到 entry state；
* 哪些普通操作允许位于配置与 consumer 之间；
* 哪个 region 构成临时状态的完整作用域；
* 未来一个 requirement 保护多个 consumer 时如何表达作用域。

raw MAD 面向硬件 emission，只表达计算和 CTRL `Consume`。临时状态的生命周期属于
guard；将来一个 requirement 保护多个 consumer 时，也不需要修改 raw op。

Verifier 规则：

1. `controlled_bits` 和 `required_bits` 均为 64-bit integer attribute，必须可静态
   比较。
2. `required_bits & ~controlled_bits == 0`。受控集合外的 required bits 必须为零，
   不能依赖“忽略未受控位”的隐式规范化。
3. `controlled_bits != 0`，空 guard 没有合法用途。
4. body 必须是单 block，且恰好包含一个实现 `StateAccessOpInterface` 的 CTRL
   `Consume` 操作；当前 MAD lowering 生成 `MadRawOp`、`MadBiasRawOp` 或 MX raw
   MAD op。
5. guard 只允许嵌套在 `func.func` 中，不能出现在最终 emission-stage IR。
6. 首版不允许 guard 嵌套，verifier 遇到嵌套时直接报错。

`RecursiveMemoryEffects` 只负责向普通优化暴露 raw MAD 对 L0A/L0B/L0C 的 memory
effects，防止 CSE/LICM 将 guard 当作无副作用操作。CTRL 状态访问仍由上述两个
state interface 描述。

### 4.4 `controlled_bits` 与 `required_bits` 的精确定义

设 guard 入口的逻辑 CTRL 为 `E`，`controlled_bits` 为 `C`，`required_bits` 为
`R`。consumer 执行时需要的硬件 CTRL 为：

```text
A = (E & ~C) | R
```

* `controlled_bits (C)`：guard 覆盖的 CTRL bit 集合。`C` 外的 bit 继承 `E`。
* `required_bits (R)`：受控 bit 中目标值为 1 的集合，必须满足
  `R & ~C == 0`。受控但不在 `R` 中的 bit 目标值为 0。

`C` 表示语义所有权，不表示运行时一定会翻转这些 bit。即使入口值已经满足
requirement，受控 bit 仍属于 `C`。优化 pass 可以据此省略不必要的 `set_ctrl`。

等价地，对任意 bit `i`：

```text
i ∉ C:  A[i] = E[i]       // 不管理，继承进入状态
i ∈ C 且 i ∈ R: A[i] = 1 // 管理并置 1
i ∈ C 且 i ∉ R: A[i] = 0 // 管理并清 0
```

`R` 必须结合 `C` 解释。目标值为 0 的 bit 表示为“在 `C` 中且不在 `R` 中”。

MAD requirement 由编译期类型和属性确定，`C/R` 使用静态 `i64` attributes。
首版仅在 `C`、`R` 分别相等时判定两个 requirement 无条件等价。不同 `(C,R)` 只有
在已知 entry state 足以证明 active state 相同时才能合并。

例如：

```text
// 只要求 CTRL[45] = 0
C = 1 << 45
R = 0

// 要求 CTRL[45] = 1，CTRL[47] = 0；不触碰 CTRL[46]
C = (1 << 45) | (1 << 47)
R = (1 << 45)
```

第二个例子不修改 CTRL[46]，并将 CTRL[47] 清零。

对 MAD，`C/R` 由 semantic op 的类型和属性静态计算：

```text
HiF8:       C |= bit45
            if (is_hif8) R |= bit45
TF32:       C |= bit46 | bit47
            if (tf32_mode is present) R |= bit46
            if (tf32_mode == round_away) R |= bit47
explicit sat/nosat:
            if (sat_mode is present) C |= bit48
            if (sat_mode == nosat) R |= bit48
n_dir:      C |= bit51
            if (has_n_dir) R |= bit51
```

HiF8 和非 HiF8 都明确设置 CTRL[45]，所以 bit45 始终在 `C` 中。`is_hif8=true`
时，它也在 `R` 中，目标值为 1；否则目标值为 0，不继承 entry state。

TF32 使用 CTRL[46:47]：bit46 控制 TF32 是否启用，bit47 选择舍入方式（0 为
`round_even`，1 为 `round_away`）。未指定 TF32 时，两位均清零。未指定
`sat_mode` 时，bit48 不受 guard 控制，继续使用 entry state 中的值；这不等同于
默认 `sat`。

典型 MAD requirement 如下。表中未列出的 CTRL bit 均不在 `C` 中，必须继承
entry state：

| MAD 语义 | `controlled_bits (C)` | `required_bits (R)` |
|---|---|---|
| 普通非 HiF8、TF32 disabled、未指定 sat、无 `n_dir` | `bit45 \| bit46 \| bit47 \| bit51` | `0` |
| HiF8，其余同上 | `bit45 \| bit46 \| bit47 \| bit51` | `bit45` |
| TF32 round-even | `bit45 \| bit46 \| bit47 \| bit51` | `bit46` |
| TF32 round-away | `bit45 \| bit46 \| bit47 \| bit51` | `bit46 \| bit47` |
| 显式 `sat` | 基础 `C \| bit48` | 基础 `R`，bit48 为 0 |
| 显式 `nosat` | 基础 `C \| bit48` | 基础 `R \| bit48` |
| 带 `n_dir` | 基础 `C`（其中包含 `bit51`） | 基础 `R \| bit51` |

“基础 `C/R`”指由 dtype、TF32 和方向等语义生成的集合。`R = 0` 表示将 `C` 中的
所有 bit 清零，不表示没有 requirement。`C = 0` 才表示不控制任何 bit，verifier
禁止这种空 guard。

consumer 在 guard body 中使用 `A`；guard 退出后的逻辑状态仍是 `E`：

```text
after guard: logical CTRL == E
```

逻辑状态恢复为 `E`，不表示 guard 末尾必须立即生成 `set_ctrl E`。后继仍需 `A`
时可以沿用当前 physical state；后继需要 `B` 时，可以直接从 `E` 构造 `B`。

接口方法实现如下：

```c++
StateResource CtrlStateGuardOp::getStateResource() {
  return StateResource::Ctrl;
}

uint64_t CtrlStateGuardOp::getControlledStateBits() {
  return static_cast<uint64_t>(getControlledBitsAttr().getInt());
}

uint64_t CtrlStateGuardOp::getRequiredStateBits() {
  return static_cast<uint64_t>(getRequiredBitsAttr().getInt());
}

Region &CtrlStateGuardOp::getGuardedBody() { return getBody(); }
```

首版要求 guard body 中的 `Consume` 同时实现 `MadRawOpInterface`。接入新的 producer
时，只需放宽 consumer verifier，不改变 guard 语义。

### 4.5 为现有 `get_ctrl/set_ctrl` 和 raw MAD 增加接口

现有 op 通过 `StateAccessOpInterface` 声明显式状态访问或隐式消费行为。

ODS 中给现有 op 增加接口声明：

```tablegen
def PTO_GetCtrlOp : PTO_Op<"get_ctrl", [
    VPTOSchedulingOpInterface,
    DeclareOpInterfaceMethods<PTO_StateAccessOpInterface>
  ]> { ... }

def PTO_SetCtrlOp : PTO_UnaryI64ConfigOp<"set_ctrl", [
    VPTOSchedulingOpInterface,
    DeclareOpInterfaceMethods<PTO_StateAccessOpInterface>
  ]>;
```

当前 `PTO_UnaryI64ConfigOp` 已支持可选 traits，无需修改 helper class。新增接口时必须
保留 `GetCtrlOp/SetCtrlOp` 现有的 `VPTOSchedulingOpInterface`。

method 实现在 `lib/PTO/IR/VPTO.cpp` 或对应的生成代码实现文件中：

```c++
StateResource GetCtrlOp::getStateResource() { return StateResource::Ctrl; }
StateAccessKind GetCtrlOp::getStateAccessKind() {
  return StateAccessKind::Query;
}
Value GetCtrlOp::getAccessedStateValue() { return getResult(); }

StateResource SetCtrlOp::getStateResource() { return StateResource::Ctrl; }
StateAccessKind SetCtrlOp::getStateAccessKind() {
  return StateAccessKind::Write;
}
Value SetCtrlOp::getAccessedStateValue() { return getValue(); }
```

四个 raw MAD 都声明 `StateAccessOpInterface`，访问语义相同：

```c++
StateResource MadRawOp::getStateResource() { return StateResource::Ctrl; }
StateAccessKind MadRawOp::getStateAccessKind() {
  return StateAccessKind::Consume;
}
Value MadRawOp::getAccessedStateValue() { return {}; }
```

`MadBiasRawOp`、`MadMxRawOp`、`MadMxBiasRawOp` 使用相同实现。interface 只声明
它们会消费 CTRL，不改变 raw MAD 的其他语义。

返回值 accessor 以 TableGen 生成代码为准。

`StateResource` 和 `StateAccessKind` 必须在生成的 op-interface 声明之前可见。
它们通过现有 enum 生成链进入 `PTOEnums.h.inc`。`PTO.h` 先包含 enums，再包含
`VPTOInterfaces.h.inc`，不需要调整 include 顺序，也不能只在 `VPTO.cpp` 中局部声明。

## 5. `VPTOExpandWrapperOpsPass` 的改动

只修改 MAD semantic-to-raw lowering 的 CTRL 部分：

```text
旧：get_ctrl -> sbitset* -> set_ctrl -> mad_raw -> set_ctrl
新：ctrl_state_guard(controlled_bits, required_bits) { mad_raw }
```

`xt` 打包、raw op 选择以及 HiF8/TF32/sat/n_dir 的语义推导保持不变。新增 helper：

```c++
FailureOr<CtrlRequirement>
buildMadCtrlRequirement(MadSemanticOpInterface op);
```

其中：

```c++
struct CtrlRequirement {
  uint64_t controlledBits;
  uint64_t requiredBits;
};
```

`controlledBits` 和 `requiredBits` 遵循第 4.4 节的定义：

```text
controlledBits = bit45 | bit46 | bit47 | optional(bit48) | bit51
requiredBits   = 根据 lhs/rhs type、tf32_mode、sat_mode、n_dir 设置
```

未指定 `sat_mode` 时，bit48 不进入 `controlledBits`。HiF8、TF32 和 `n_dir`
始终控制各自字段，并根据类型或属性设置 `requiredBits`。

`lowerMadSemanticOp()` 的 CTRL 部分调整为：

```c++
FailureOr<CtrlRequirement> requirement = buildMadCtrlRequirement(op);
if (failed(requirement))
  return failure();

auto guard = rewriter.create<pto::CtrlStateGuardOp>(
    loc, requirement->controlledBits, requirement->requiredBits);

// 在 guard body 中保留现有 raw-op 选择和 xt 打包结果。
OpBuilder::InsertionGuard insertionGuard(rewriter);
rewriter.setInsertionPointToStart(&guard.getBody().emplaceBlock());
emitMadRawOp(op, deriveMadRawKind(op), xt, rewriter);
rewriter.eraseOp(op);
```

builder 签名以生成的 ODS API 为准。改动后，该 pass 只计算静态 requirement 并
创建 guard，不再生成 `GetCtrlOp`、`SetCtrlOp` 或 `Sbitset*Op`。entry 的保存、配置
共享和状态恢复交给 `VPTOOptimizeCtrlStatePass`。

AccStore、atomic 和 MASK lowering 保持不变；它们的独立 `get_ctrl/set_ctrl` 不与
MAD guard 合并。

## 6. `VPTOOptimizeCtrlStatePass` 的优化与物化

该 pass 先分析状态、生成物化计划，再统一修改 IR。本节先定义相关术语。

### 6.1 核心概念与状态表示

#### Requirement、entry 与 active

guard requirement 为 `(C,R)`。`C` 是受控 CTRL bits，`R` 是其中目标值为 1 的
集合。入口逻辑状态为 `E` 时，consumer 所需的完整状态如下：

```text
active(E,C,R) = (E & ~C) | R
```

`entry` 指 guard 入口处的 logical state，`active` 指 guard body 执行时安装在硬件
中的完整状态。`C` 外的 bits 从 entry 继承。

#### Logical state 与 physical state

pass 在每个程序点维护两种状态：

* `logical`：按 guard 作用域语义，显式 `get_ctrl` 在此处应观察到的 CTRL。进入和
  退出临时 guard 都不改变 logical；独立 `set_ctrl` 会更新 logical。
* `physical`：当前已知安装在硬件 CTRL 中的状态。guard 退出后，physical 可以暂时
  保持 active，而 logical 仍为 entry。

逻辑状态恢复不一定立即生成物理 `set_ctrl`。遇到显式观察点、状态边界或函数出口
时，pass 必须恢复 physical，或者证明 physical 已经与 logical 等价。

状态表示如下：

```c++
struct AbstractCtrlValue {
  // Null when no exact full-register SSA value is available.
  Value fullValue;

  // Per-bit facts valid even when fullValue is null.
  APInt knownBits;
  APInt knownValues;
};

struct CtrlState {
  // The value an explicit get_ctrl is required to observe.
  AbstractCtrlValue logical;

  // The value known to be installed in hardware at the current point.
  AbstractCtrlValue physical;

  // True when physical and logical are equal even if their value is unknown.
  bool physicalMatchesLogical;
};
```

除非函数入口有可信的完整 CTRL 定义，否则初始状态均为 unknown；不能假设 kernel
入口 CTRL 为零。`fullValue` 保存完整 SSA base，`knownBits/knownValues` 保存逐 bit
信息。`physicalMatchesLogical` 记录二者的关系，不表示具体 bit 值。函数入口处即使
CTRL 内容未知，该字段也为 true；安装临时 active state 后，除非能证明 active 与
logical 相同，否则设为 false。

#### 优化、物化、透明操作与状态边界

“优化”包括选择 CTRL 切换位置、共享 entry 读取、合并重复配置和延迟恢复；
“物化”包括生成 `get_ctrl/sbitset*/set_ctrl`、展开 guard body 和删除 guard。

`CTRL-transparent` 操作不读、写或隐式消费 CTRL，也不改变 logical/physical。
状态边界包括独立 `get_ctrl/set_ctrl`、未知 call、inline asm、未声明 requirement
的 CTRL consumer 和 `Clobber`。边界前按需恢复状态，边界后更新或清空状态信息。

### 6.2 Pass 输入、输出与执行阶段

pass 的输入包括：

* `CtrlStateGuardOp`；
* 实现 `StateAccessOpInterface` 的 `GetCtrlOp/SetCtrlOp` 和 raw MAD；
* 普通 VPTO 操作和控制流。

pass 每次执行包含两个步骤：

1. 分析步骤在包含 block edge 和结构化 region edge 的状态流图上传播
   logical/physical 状态，为每个 guard 和状态边界建立 `MaterializationPlan`，
   不修改 IR。
2. 物化步骤应用完整 plan，生成必要的 CTRL SSA 计算和硬件读写，展开 guard body，
   删除全部 `CtrlStateGuardOp`。

输出必须满足以下条件：

* 不存在 `CtrlStateGuardOp`；
* guard 物化只生成语义必要的 `GetCtrlOp/SetCtrlOp`；
* `sbitset*` 和算术表达式只在需要构造新的 active CTRL 时出现；
* 未参与 guard 优化的现有独立 `get_ctrl/set_ctrl` 保持原顺序。

### 6.3 优化区间与成立条件

#### 候选区间

pass 扫描 `CtrlStateGuardOp` 及其间的操作，构造可共享 logical base、延迟恢复的
候选区间：

```text
guard A -> transparent/compatible operations -> guard B -> guard C
```

两个 guard 要位于同一候选区间，必须满足：

1. guard 位于同一 basic block，或跨越受支持的 SCF region；跨 region 时必须分析
   所有能到达后一个 guard 的控制流路径，且各路径状态合流后仍能确定优化所需的
   logical/physical 状态；
2. 所有相关路径上都没有本版本不改写的独立 `get_ctrl/set_ctrl`；
3. 所有相关路径上都没有未知 call、inline asm 或未声明的 CTRL clobber；
4. guard 没有不受支持的嵌套区域；
5. requirement 是静态可比较的，且所需 logical/physical facts 可证明。

跨 SCF region 时，不能只检查前一个 guard 到后一个 guard 的某一条路径。如果存在
绕过前一个 guard 的路径，或任一入口路径经过状态边界，合流后的 physical state
通常为 unknown，两个 guard 不能共享已有配置。首版不跨已有 CF edge 建立候选区间；
按需求增加 CF successor 分析后，再复用相同规则。

任一条件不满足，候选区间就在该处结束，并物化必要的恢复操作。同一区间可以共享
logical base 或省略中间 restore，但不同 requirement 仍可能需要写入新的 active state。

#### 普通操作的 opt-in 透明规则

guard 之间的普通操作采用 opt-in 透明规则：

* `arith.constant` 以及经审核不访问隐式硬件状态的纯 SSA 操作可标记为
  CTRL-transparent；
* 实现 `StateAccessOpInterface` 的操作不是透明操作，按 access kind 处理；
* 未经审核的 PTO op、未知 dialect op 和有 region 的未知 op 默认不是透明操作；
* pass 中使用集中维护的 `isCtrlTransparent(Operation *)` helper，不允许在各个
  rewrite pattern 中散落 op-name 特判。

透明性只能由集中白名单或显式标记授予，不能因为操作未实现
`StateAccessOpInterface` 就判定为透明。对于未声明 CTRL 访问语义、也未加入透明
操作白名单的 op，pass 将其作为状态边界：在该处结束当前候选区间，按需恢复
logical state，并在操作后重新开始状态分析。这可能少合并一些 guard，但不会让临时
CTRL 配置越过语义未知的操作。

例如：

```text
guard A { mad_raw_0 }
%v = arith.addi %x, %y : i64       // 已审核为 CTRL-transparent
guard A { mad_raw_1 }
```

`arith.addi` 不依赖 CTRL，因此 physical 可以保持 active A，两个 MAD 之间无需恢复
entry 或再次 `set_ctrl`。

#### 会访问 CTRL 的中间操作

透明是保持 active 的充分条件，但不是必要条件。CTRL 访问按接口语义处理：

* 显式 `get_ctrl` 是 `Query`，必须观察 logical state。如果 physical 仍是临时
  active，且与 logical 不同，必须先恢复。
* 另一个带已知 `(C,R)` 的 guard 可以直接使用当前 physical，条件是 pass 证明：

  ```text
  physical == (logical & ~C) | R
  ```

  只证明 `(physical & C) == R` 还不够；`C` 外的位也必须等于 logical，因为这些位
  按 guard 语义从 entry 继承。只有完整目标状态等价时，不同 `(C,R)` 才能省略新的
  `set_ctrl`。
* 未声明 requirement 的 CTRL `Consume` 无法证明兼容，首版先恢复 logical，
  并把该操作作为候选区间边界。
* 独立 `set_ctrl` 是 `Write`。首版先恢复 logical，保留原 write，再将
  logical/physical 更新为它的 operand，避免间接改写尚未接入 guard 的 lowering。
* 未知 call、inline asm 或 `Clobber` 前按语义恢复；执行后 logical 和 physical 的
  具体值均为 unknown，但两者仍表示同一个调用后状态，
  `physicalMatchesLogical` 设为 true。

读取或依赖 CTRL 不一定触发恢复。只要访问语义已建模，且 physical 满足该操作应
观察或消费的完整状态，就可以沿用 active state。

### 6.4 单个 guard 的物化

对 guard `(C,R)`：

1. 若 `physical` 已经等价于 `(logical & ~C) | R`，不生成写入。
2. 若存在当前位置可用的完整 `logical.fullValue`，直接用它构造 active value。
3. 若没有 `logical.fullValue`，只有在 `physicalMatchesLogical == true` 时才能生成
   `get_ctrl`，并将读取结果作为 logical base。
4. 若 `physicalMatchesLogical == false` 且没有 `logical.fullValue`，当前物化计划不
   成立。pass 必须在 physical 第一次偏离 logical 前安排 `get_ctrl`，不能在当前位置
   把临时 active state 读作 logical base。
5. 只对 `C` 中的 bits 生成 `sbitset0/sbitset1` 或等价的位更新。
6. 只有 active value 与 `physical` 不等价时才生成 `set_ctrl active`。
7. body 执行不改变 `logical`。将 `physical` 记录为 active state，是否恢复由后续
   操作决定。

### 6.5 连续 guard 的优化

输入：

```text
guard C,R { mad_raw_0 }
guard C,R { mad_raw_1 }
guard C,R { mad_raw_2 }
```

物化为：

```text
%entry = pto.get_ctrl
%active = update_bits(%entry, C, R)
pto.set_ctrl %active
pto.mad_raw_0
pto.mad_raw_1
pto.mad_raw_2
```

只有后继需要 `entry` 时才生成恢复写入；需要另一组 `(C2,R2)` 时，直接从 `%entry`
计算：

```text
%active2 = update_bits(%entry, C2, R2)
pto.set_ctrl %active2
```

两者之间不需要先执行 `pto.set_ctrl %entry`。

### 6.6 非完全覆盖的 requirement

第一个 guard 只覆盖 bit45，第二个 guard 只覆盖 bit47：

```text
A1 = (entry & ~bit45) | V1
A2 = (entry & ~bit47) | V2
```

第二个 guard 不控制 bit45，因此 `A2[45]` 必须等于 `entry[45]`。第一个 guard 可能
临时修改了该位。例如 `entry[45] = 1`、`V1 = 0` 时，`A1[45] = 0`。如果只在 `A1`
上更新 bit47，得到的值仍满足 `A2[45] = 0`，不符合第二个 guard 继承
`entry[45]` 的语义。

默认应以同一个 logical entry 分别构造两个 active value：

```text
%A1 = update_bits(%entry, bit45, V1)
pto.set_ctrl %A1
mad_raw_1

%A2 = update_bits(%entry, bit47, V2)
pto.set_ctrl %A2
mad_raw_2
```

这里复用 `%entry` 作为 SSA 计算的 base，不要求在两个 guard 之间先执行
`pto.set_ctrl %entry`。只有能够证明 `A1[45] == entry[45]` 时，才可以直接以 `A1`
为 base 构造 `A2`。

### 6.7 显式访问和独立访问的处理

#### 显式 `get_ctrl`

```text
guard C,R { mad_raw }
%x = pto.get_ctrl
```

在 `%x` 前必须把硬件状态恢复到 guard 进入时的 logical state；`%x` 不能读到 MAD
的临时 active state。

首版保留输入 IR 中已有的 `get_ctrl`，不以 logical SSA 替换。query result 作为新的
logical/physical SSA base，避免改写尚未接入 guard 的现有序列。

#### 独立 `set_ctrl`

独立 `set_ctrl` 在首版中是优化边界：

1. 按需恢复当前 guard 的 logical state；
2. 保留原有的独立 `set_ctrl`；
3. 将新的 physical/logical state 设为该 operand，之后重新开始候选区间。

#### 未知调用

未知 `func.call`、inline asm 或 backend op 可能修改 CTRL：

```text
guard C,R { mad_raw }
func.call @unknown()
guard C,R { mad_raw }
```

调用前恢复 logical。调用后，logical 和 physical 的具体值均为 unknown，但
`physicalMatchesLogical` 为 true；第二个 guard 因此可以重新读取实际 CTRL，再构造
active state。只有可靠的 callee state summary 才能放宽此规则。

#### Transfer table

| 输入操作 | 操作前 | 操作本身 | 操作后状态 |
|---|---|---|---|
| 预先存在的 `Query` | 若 physical != logical，先恢复 logical | 原样保留 query | logical = physical = query result |
| 预先存在的完整 `Write` | 若 physical != logical，先恢复 logical | 原样保留 write | logical = physical = write operand |
| guard 外的 `Consume` | 若 physical != logical，先恢复 logical | 原样保留 consumer | logical = physical，结束候选区间 |
| `Clobber` | 若语义要求，先恢复 logical | 原样保留 clobber | logical/physical 均为 unknown，且两者相等 |
| CTRL-transparent op | 不物化恢复 | 原样保留 | 状态不变 |
| `func.return` | 必须恢复 logical | 原样保留 return | 函数边界不泄漏临时状态 |

“预先存在”指 pass 输入 IR 中已有的访问。首版只优化 guard 物化产生的 query/write，
不改写这些独立访问。

### 6.8 控制流、分支与循环（实现阶段 2）

这里的阶段编号对应第 9 章的开发顺序。实现阶段 1 是便于开发和测试的中间状态；首个
可交付版本必须继续完成实现阶段 2，支持 `scf.if`、`scf.for` 和嵌套循环。

实现阶段 1 会递归访问嵌套 region 中的 block，但每个 basic block 独立分析。它只合并
同一 block 内的 guard，不把状态传播到后继 block 或跨越 region 边界。离开当前分析
区间前按需恢复 logical state，因此不处理分支合流、循环回边和循环外提。

实现阶段 2 在同一个 pass 中增加跨 block、跨 region 的前向数据流分析。该位置尚未执行
SCF-to-CF，正常 VPTO 控制流仍以 `scf.if` 和 `scf.for` 表示。首版通过
`RegionBranchOpInterface` 建立这两类 op 的 region 入口和出口状态边，再通过
`LoopLikeOpInterface` 识别 `scf.for`，生成循环摘要并选择配置位置。

输入中也可能已经存在 `cf.br/cf.cond_br`，但它们不是本阶段由 SCF lowering 生成的。
这类 CF 控制流可按需求增加兼容：通过 basic block successor edge 传播状态，并复用
本节的 meet 规则。该能力不属于首版实现范围。首版遇到已有 CF edge 时，在 terminator
前按需恢复 logical state，后继 block 以 unknown 具体值且
`physicalMatchesLogical == true` 重新开始分析。其他未支持的 region op 同样作为状态
边界处理。

#### 6.8.1 Block 状态与合流

每个 block 保存入口和出口 `CtrlState`。block transfer 按第 6.7 节的规则顺序处理
操作。首版沿受支持的 SCF region edge 传播状态；增加 CF 兼容后，再沿 basic block
successor edge 传播。一个状态点有多个前驱时，对所有前驱状态执行 meet：

```text
所有前驱都知道某个 bit，且值相同 -> 该 bit 保持 Known
任一前驱为 unknown，或已知值不同    -> 该 bit 变为 Unknown

所有前驱的 fullValue 是同一个可用 SSA value -> 保留 fullValue
否则                                      -> fullValue 为空

所有前驱都满足 physicalMatchesLogical -> 合流后仍为 true
否则                                   -> 合流后为 false
```

`physicalMatchesLogical` 表示逐路径关系。即使不同前驱上的 CTRL 具体值不同，只要每条
路径都满足 physical 等于各自的 logical，合流后该关系仍成立；此时可以在合流点通过
`get_ctrl` 获取实际到达路径上的 logical value。

不能只采用某一个前驱的状态。例如一条路径保持 active A，另一条路径保持 logical E，
合流后不能假定 physical 为 A。后继 guard 必须根据 meet 结果重新判断是否需要配置。

#### 6.8.2 分支

首版分析 `scf.if` 时，then/else 使用相同的入口状态，各自计算出口状态，再在
`scf.if` 之后执行 meet：

```text
                entry state
                 /        \
          then transfer  else transfer
                 \        /
                  meet state
```

只有所有可达分支在合流点都得到兼容的 physical state，后继 guard 才能沿用现有配置。
任一分支经过独立 `get_ctrl/set_ctrl`、未知 call、inline asm 或 `Clobber`，都会降低或
清空合流后的状态信息。没有 else region 的 `scf.if` 必须把“不执行 then”的路径也
纳入 meet。

按需求支持已有 `cf.cond_br` 时，true/false successor 使用相同规则；`cf.br` 只传播
当前 block 的出口状态。CF 兼容不能通过先运行 SCF-to-CF 实现，因为这会丢失
`scf.for` 的结构化循环信息，不利于嵌套循环摘要和配置外提。

#### 6.8.3 单层循环

循环 header 同时接收 loop entry 和 backedge 状态。实现阶段 2 反复计算下面的关系，直到
`CtrlState` 不再变化：

```text
headerState = meet(loopEntryState, backedgeState)
backedgeState = transfer(loopBody, headerState)
```

循环所需的 active state 稳定，必须同时满足：

1. 每次迭代的 `(C,R)` 相同，或能证明完整 active state 等价；
2. logical entry `E` 在循环内不变；
3. 所有到达 backedge 的路径都以同一个 active state 结束；
4. 循环体内没有独立 `get_ctrl/set_ctrl`、未知 call、inline asm、`Clobber`，也没有
   要求另一完整 CTRL 状态的 consumer；
5. active state 的 SSA 计算不依赖 induction variable 或其他 loop-variant value；
6. 所有 loop exit 上都能保留或按需恢复 logical state。

满足这些条件时，可以在 loop entry 生成一次配置，并让所有迭代沿用：

```text
// loop entry
%entry = pto.get_ctrl
%active = update_bits(%entry, C, R)
pto.set_ctrl %active

scf.for ... {
  pto.mad_raw ...
  // backedge 上 physical 仍为 %active
}

// 仅在后继需要 logical 时生成
pto.set_ctrl %entry
```

恢复仍可延迟到循环后的第一个观察点或状态边界，不要求紧跟在 loop 后面。

若循环可能执行零次，不能无条件把有状态副作用的 `set_ctrl` 移到普通 preheader。
首版只对静态证明 trip count 大于零的循环执行外提；动态零次循环仍在原 guard 位置
物化。后续若要支持，可在确认进入 loop body 的控制流边上建立专用 loop-entry block，
但不能无条件执行配置。

无法证明 loop-carried state 稳定时，不跨 backedge 复用 physical state。pass 仍可在
单次迭代内合并相邻 guard，但最终配置保留在原 guard 附近，并在 pass 结束前删除所有
`CtrlStateGuardOp`。“不跨 backedge”不表示最终 IR 中保留 guard。

#### 6.8.4 嵌套循环

嵌套循环必须纳入实现阶段 2。issue #1279 的 GEMM 在 K-tile 循环内重复执行多次 MAD；
实际 IR 还可能在 K-loop 外包含 M/N tile 循环。只做单 block 优化，最多把一次迭代内
多个 MAD 的配置合并为一次，仍会在每次 K 迭代重复配置。

循环按从内到外的顺序分析。内层循环达到不动点后，生成状态摘要，至少包括：

```text
入口 logical/physical 要求
是否保持 logical
循环出口的 physical 状态
是否包含状态边界
是否可证明至少执行一次
```

外层循环把内层循环作为一个带摘要的状态转移点，而不是默认的 unknown region。配置
放在能够满足全部路径条件的最外层 loop entry：

```text
outer M/N loop {
  inner K loop {
    guard A { mad_raw_0 }
    guard A { mad_raw_1 }
  }
}
```

处理顺序为：

1. 在 inner K-loop 内合并相同 active requirement；
2. 若 inner loop 的 backedge 状态稳定，将配置移到 inner loop entry；
3. 若 outer loop 的其他操作也不观察或改变 CTRL，且跨越的每一层循环都能证明
   trip count 大于零，可继续将配置移到 outer loop entry；
4. 在最高合法作用域之后的第一个观察点或状态边界恢复 logical state。

如果 inner loop 前后存在需要 logical state 的操作、另一组 CTRL requirement 或未知
调用，配置只能停留在 inner loop entry。不同嵌套层级分别做合法性判断，不能因为
inner loop 稳定就直接外提到最外层循环。

#### 6.8.5 实现阶段 2 的保守退出条件

出现以下任一情况时，停止跨 edge 或跨 loop 优化，并在局部完成物化：

* block/region 的状态 meet 丢失了构造 active state 所需的信息；
* region op 没有受支持的控制流语义；
* 循环存在无法建模的 exit、异常边或动态嵌套控制流；
* zero-trip 路径会使配置被提前到本不执行 guard 的路径；
* active state 或 logical base 依赖 loop-variant value；
* 任一路径经过未建模的 CTRL 访问。

保守退出只减少优化机会，不改变 guard 的临时状态语义。

### 6.9 Pass 伪代码

下面给出实现阶段 2 的首版流程。`StateFlowGraph` 包含受支持的 SCF region branch edge
和 loop backedge；循环按 post-order 收集，保证先得到内层循环摘要。后续兼容已有
CF 时，再把 basic block successor edge 加入同一张图。

```c++
runOnFunc(func) {
  auto guards = collectCtrlStateGuards(func);
  StateFlowGraph graph = buildStateFlowGraph(
      func, /*supportedRegions=*/{scf::IfOp, scf::ForOp});

  // Includes SCF region entry/exit edges and loop backedges.
  StateSolution states = solveCtrlStateFixpoint(graph, guards);

  // Inner loops are summarized before their parents.
  DenseMap<Operation *, LoopCtrlSummary> loopSummaries;
  for (LoopLikeOpInterface loop : collectLoopsInPostOrder(func))
    loopSummaries[loop] = summarizeLoop(loop, states, loopSummaries);

  MaterializationPlan plan;
  planRegionsRecursively(func, states, loopSummaries, plan);
  planLoopPlacements(collectLoopsInPostOrder(func), states,
                     loopSummaries, plan);

  applyMaterializationPlan(plan);
  assertNoCtrlStateGuardsRemain(func);
}
```

不动点计算、循环摘要和 plan 构建都不修改 IR。`planRegionsRecursively` 按状态解遍历
嵌套 region，并将 guard 及其 raw MAD 作为一个原子 consumer；
`planLoopPlacements` 再根据 inner-to-outer 摘要选择最高合法配置位置。所有合流和循环
信息稳定后才统一物化，避免边分析边重写导致状态依据失效。

## 7. 正确性不变量

优化前后必须满足：

1. 每个 MAD raw consumer 执行时，`controlled_bits` 中的 CTRL bits 与原程序一致。
2. `controlled_bits` 外的 CTRL bits 保持进入 guard 时的逻辑值。
3. guard 外显式 `get_ctrl` 观察到的值与未优化程序一致。
4. 独立 `set_ctrl` 的顺序和可观察语义保持不变。
5. 未知调用前后不假设 CTRL 保持不变。
6. 函数入口、return 和 kernel 边界不泄漏 MAD 临时配置。
7. physical 第一次偏离 logical 前，必须保留可用于恢复和构造后续 active state 的
   完整 logical SSA value。
8. 原程序不执行 guard 的 zero-trip 路径，不得因循环外提而执行临时 CTRL 配置。
9. 每个 loop exit 上的 physical state 必须等于 logical state，或由后续状态分析继续
   跟踪并在第一个观察点前恢复。
10. 无法证明等价时保留原始 `get_ctrl/set_ctrl`。

## 8. 测试计划

测试只覆盖 CTRL guard 和 get/set 优化，不测试 AccStore、MASK 或 atomic 优化。

### 8.1 IR/FileCheck

新增 `test/lit/vpto/ctrl/` 测试：

* 三个相同 MAD guard 只产生一次配置写入；
* HiF8/普通 FP8 交替时 bit45 正确切换；
* TF32、显式 sat/nosat、n_dir 的 controlled/required bits 正确；
* 未指定 sat 时 bit48 从 logical state 继承；
* 两组不同 requirement 之间直接切换，不产生无意义的 restore；
* physical 保持前一个 active state 时，复用已保存的 logical value，不重新执行
  `get_ctrl` 读取临时状态；
* `scf.if` 两侧状态相同时，合流后沿用配置；状态不同时，在合流点重新配置；
* 静态正 trip count 的 K-loop 将配置移到 loop entry，循环体内不再出现 CTRL 读写；
* 可能 zero-trip 的循环不把 `set_ctrl` 无条件移到 preheader；
* 嵌套 M/N-loop 与 K-loop 状态均稳定时，将配置移到最高合法 loop entry；
* 内外层循环之间存在状态边界时，只在 inner loop entry 配置；
* 显式 `get_ctrl` 前恢复 logical state；
* 预先存在的独立 `get_ctrl/set_ctrl` 作为边界，操作本身原样保留；
* guard 外的 raw MAD `Consume` 作为边界；
* 未知 call 后重新读取 CTRL；
* 内部 `ctrl_state_guard` 不出现在 emission-stage IR；
* `--emit-vpto` 中最终 `get_ctrl/set_ctrl` 数量符合预期。

### 8.2 性能验证

除 issue 中的 GEMM 外，至少比较以下场景：

```text
重复相同 MAD 的长 K 循环
外层 M/N tile、内层 K tile 的嵌套循环
动态 zero-trip 循环
不同 MAD requirement 交替
短 K kernel（确认固定开销没有被放大）
无 MAD guard 的 kernel（确认不改变独立 get/set）
```

统计 `get_ctrl`、`set_ctrl`、`sbitset` 数量、循环体 scalar issue、AIC cycle 和
端到端 latency。数值结果必须与 baseline 一致。

## 9. 分阶段实现

以下阶段表示开发顺序。实现阶段 1 可独立编译和测试，但不是 issue #1279 的完整交付；
首版交付包含实现阶段 1 和实现阶段 2。实现阶段 3 是后续扩展，不在本次范围内。

### 实现阶段 1：内部 guard 与 basic-block 优化

1. 增加 `StateAccessOpInterface`、`StateGuardOpInterface` 和 `CtrlStateGuardOp`。
2. 让 MAD semantic-to-raw 生成 guard，不再立即生成保存/配置/恢复序列。
3. 实现 `VPTOOptimizeCtrlStatePass` 的同一 basic block forward scan：共享
   logical base、删除重复 write、延迟 restore。
4. 增加 emission-stage verifier，禁止 guard 残留。

### 实现阶段 2：控制流与循环

1. 在同一个 `VPTOOptimizeCtrlStatePass` 中加入 block 和 region entry/exit fixpoint。
2. 通过 `RegionBranchOpInterface` 处理 `scf.if` 的分支合流。
3. 通过 `LoopLikeOpInterface` 处理 `scf.for` 的 backedge、zero-trip 和 exit 状态。
4. 按 inner-to-outer 顺序生成循环摘要，支持 issue #1279 所需的嵌套循环分析。
5. 只对静态正 trip count 且状态稳定的循环执行配置外提。
6. 保持未知 region、状态边界和不稳定回边的保守规则。
7. 按需求增加已有 `cf.br/cf.cond_br` 的 block successor edge 分析，不在首版实现。

### 实现阶段 3：扩展其他使用者

如果以后发现 AccStore、MASK 或其他特殊状态也存在同类往返：

1. 如果它们仍然读写同一个硬件 CTRL，必须复用 `StateResource::Ctrl` 和
   `CtrlStateGuardOp`，只新增 requirement producer 并放宽 body consumer verifier；
2. 只有目标是另一个物理特殊寄存器时，才新增 `StateResource` 和对应 guard op；
3. 非静态 64-bit 部分覆盖需先扩展 guard interface 和状态表示；
4. 扩展状态转移和测试，不复制新的专用 pass。

实现阶段 3 不属于首版范围。

## 10. 文件改动清单

| 文件 | 改动 |
|---|---|
| `include/PTO/IR/PTOAttrs.td` | 新增 `StateResource`、`StateAccessKind` enum |
| `include/PTO/IR/VPTOInterfaces.td` | 新增两个 state interface |
| `include/PTO/IR/VPTOOps.td` | 新增 `CtrlStateGuardOp`；为 get/set 和 raw MAD 声明 interface，并保留现有 scheduling interface |
| `lib/PTO/IR/VPTO.cpp` | 实现 resource、access kind 和 operand/result 映射 |
| `lib/PTO/Transforms/VPTOExpandWrapperOps.cpp` | 将 MAD CTRL lowering 改为生成 guard |
| `lib/PTO/Transforms/VPTOOptimizeCtrlState.cpp` | 新增 CTRL 状态优化和物化 pass |
| `include/PTO/Transforms/Passes.td`、`Passes.h`、`lib/PTO/Transforms/CMakeLists.txt` | 注册并构建新 pass |
| `tools/ptoas/ptoas.cpp` | 在 emission validation 前加入新 pass |
| `lib/PTO/Transforms/PTOValidateVPTOIR.cpp` | authoring stage 接受 guard，emission stage 拒绝 guard |
| `test/lit/vpto/ctrl/` | 新增 lowering、正确性和冗余消除测试 |
