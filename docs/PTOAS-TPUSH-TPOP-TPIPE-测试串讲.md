# PTOAS TPUSH / TPOP / TPIPE 接口测试串讲

> 内容来源：
> - `PTOAS/docs/HL_ptoisa_newfeature20260307_TPUSH_TPOP.md`（硬件 / ISA 背景）
> - `PTOAS/docs/designs/ptoas-tpush-tpop-design.md`（PTOAS 前端 IR 和内部 IR 设计）
> - `PTOAS/include/PTO/IR/PTOOps.td`（仓上最新 op 定义）

---

## 1. 背景

### 1.1 这套指令解决什么问题

`ExpandMixedKernel` pass 会把一个混合的 InCore function 拆成多个共调度的子 kernel——常见做法是把数据搬运交给 Vector、把计算交给 Cube。这两个子 kernel 同处一个 cluster，需要一条带同步、能流水化的数据通道。

TPUSH / TPOP / TPIPE 就是这条通道的 ISA 抽象：

- TPIPE：producer → consumer 的一条多 slot 环形 FIFO。
- TPUSH：producer 把一个 tile 放进 FIFO 的一个 slot。
- TPOP：consumer 拿到下一个就绪 slot 里的数据。
- TFREE：consumer 显式释放 slot，让 producer 可以覆盖。

### 1.2 数据通路：Cube 和 Vector 之间的数据怎么走

cluster 内 Cube 和 Vector 互相传 tile 时，数据先落到一段环形 FIFO，再被 consumer 取走。FIFO 的物理位置在两类平台上不一样：

| 平台 | FIFO 落在哪里 |
|---|---|
| A2 / A3 | GM（off-chip 全局内存），cluster 内所有 core 都能访问 |
| A5 | consumer 自己的 SRAM——consumer 是 Vector 就落在 Vector 的 UB，consumer 是 Cube 就落在 Cube 的 L1 |

```
A2 / A3：                                  A5：

Producer    GM           Consumer         Producer                Consumer
┌──────┐ ┌──────────┐  ┌──────┐          ┌──────┐             ┌────────────┐
│      │→│slot[0..N]│→│      │           │      │   DMA 到    │ UB / L1    │
│Cube/V│ │(off-chip)│  │V/Cube│          │Cube/V│  consumer   │slot[0..N]  │
└──────┘ └──────────┘  └──────┘          └──────┘   本地 SRAM  │(on-chip)   │
                                                              └────────────┘
```

A5 把 FIFO 放进 consumer 本地 SRAM 后，consumer 不用再 TLOAD 一次，直接零拷贝读，握手延迟更低。

### 1.3 一卡内的 Cube 和 Vector：拓扑与同步

一个 cluster 包含 1 个 Cube core 和 2 个与之配对的 Vector core（文档里叫 buddy Vector）。它们之间靠硬件 flag 做同步：

```
┌─────────────────── Cluster ───────────────────┐
│                                                │
│  ┌────────┐   每方向 8 个 flag    ┌────────┐  │
│  │ Vector0│◄──────────────────────►│        │  │
│  └────────┘   V→C / C→V 各 8 个    │  Cube  │  │
│                                    │        │  │
│  ┌────────┐                        │        │  │
│  │ Vector1│◄──────────────────────►│        │  │
│  └────────┘                        └────────┘  │
└────────────────────────────────────────────────┘
```

- 每一对 Vector–Cube 在 V→C、C→V 两个方向上各有 8 个 flag，单对总共 16 个。
- 一个 cluster 有 2 对 Vector–Cube，所以跨核 flag 总共 32 个。
- 一个 InCore function 只跟它对面的一个 buddy 通信，所以用到的 flag 在 16 个以内。

Vector 一侧 SET 一个 flag、Cube 一侧 WAIT，就完成一次"数据已就绪"或"slot 已空闲"的握手；反过来也一样。

### 1.4 谁是 producer、谁是 consumer 不固定

producer 和 consumer 是逻辑角色，由具体场景决定：

- Cube 做 producer 的典型场景：matmul 算完交给 Vector 后处理。
- Cube 做 consumer 的典型场景：接 Vector 预处理好的数据。
- Vector 同理。
- 双向场景下，Cube 和 Vector 同时既当 producer 又当 consumer，对应两条反向 FIFO。

### 1.5 SLOT_NUM（FIFO 多少个槽）

SLOT_NUM 由方向决定，跟下面 §1.6 说的 `split` 没关系：

| 通信模式 | DIR_MASK | SLOT_NUM | 占用 flag |
|---|---|---|---|
| 单向（只 C2V 或只 V2C） | 1 或 2 | 8 | 8 |
| 双向（C2V + V2C 同时） | 3 | 每方向 4 | 4 + 4 |

直观理解：单向时一个方向独占全部 8 个 flag、跑得最深；双向时两个方向各分一半。

### 1.6 split 与 nosplit

数据传输指令上的 `split` 属性取三个值：

- `TILE_NO_SPLIT (0)`：sub-core 之间不切 tile。
- `TILE_UP_DOWN (1)`：上下切。
- `TILE_LEFT_RIGHT (2)`：左右切。

`split` 控制底层 TALLOC / TPUSH / TPOP / TFREE 怎么执行：tile entry 上决定 sub-core 间怎么切 tile，global entry 上决定 GM slot 的 sub-core offset 怎么算。

PTOAS 自己不解释 `split` 的语义，只校验枚举合法、把值透传到底层。

初始化 op 上还有一个可选的 `nosplit` 布尔属性，用来声明该 pipe 是否走"不切"路径：

- `nosplit = true`：该 pipe 上所有数据传输 op 必须用 `split = 0`。
- `nosplit = false`：该 pipe 上所有数据传输 op 必须用 `split = 1` 或 `2`。
- 不写：由 `pto-infer-validate-pipe-init` pass 根据下游 op 的 split 推断出来。

`nosplit` 在两个平台上启用的协作模式不同：

- A5：`nosplit = true` 走 `1C : 1V`——pipe 序列只在一个 Vector core 上跑。
- A2 / A3：硬件强制 `1C : 2V`，两个 Vector core 必须跑相同代码，并且对同一条 pipe 上的 `talloc` / `tpush` / `tpop` / `tfree` 保持完全相同的发出顺序（两边时间点可以错开，相对顺序要一致）。

---

## 2. 方案

PTOAS 把这套接口分两层：

- 前端 IR：用户和测试用例直接写的接口。
- PTOAS 内部 IR：前端 lowering 之后的中间形态，最终由 EmitC 翻译成 pto-isa C++。

### 2.1 前端有哪些 op

数据通信相关的前端 op 一共 10 个：

| 类别 | op | 跑在哪 | 角色 |
|---|---|---|---|
| 初始化 | `pto.aic_initialize_pipe` | Cube | 绑定 pipe，并在 Cube 当 consumer 的方向上预先 SET free flag |
| 初始化 | `pto.aiv_initialize_pipe` | Vector | 同上，但是 Vector 当 consumer 时预设 |
| 申请 GM slot | `pto.talloc_to_aiv` | Cube | C2V，仅 global entry 用 |
| 申请 GM slot | `pto.talloc_to_aic` | Vector | V2C，仅 global entry 用 |
| 提交 | `pto.tpush_to_aiv` | Cube | C2V，tile / global entry 都支持 |
| 提交 | `pto.tpush_to_aic` | Vector | V2C，tile / global entry 都支持 |
| 取数据 | `pto.tpop_from_aic` | Vector | C2V，wait ready + 拿数据，不释放 slot |
| 取数据 | `pto.tpop_from_aiv` | Cube | V2C，wait ready + 拿数据，不释放 slot |
| 释放 | `pto.tfree_from_aic` | Vector | C2V，与 `tpop_from_aic` 配对 |
| 释放 | `pto.tfree_from_aiv` | Cube | V2C，与 `tpop_from_aiv` 配对 |

初始化 op 的常用属性：

- `id`：本函数内逻辑 pipe 的编号，数据传输 op 通过 `id` 找到自己绑定的 pipe。
- `dir_mask`：方向，1 / 2 / 3。
- `slot_size`：单个 entry 字节数（切分前的完整大小）。
- `local_slot_num`：可选，A2/A3 tile entry 路径上覆盖 consumer 侧 local FIFO 槽数。
- `nosplit`：可选 bool，约束该 pipe 上的所有传输 op 走 split=0（true）或 split=1/2（false）。
- `gm_slot_buffer` / `gm_slot_tensor` / `c2v_consumer_buf` / `v2c_consumer_buf`：地址相关，按平台和 entry 类型挑选。

A5 上 tile entry 路径还会用到两个地址相关的 op：

| op | 干什么 |
|---|---|
| `pto.reserve_buffer` | 在当前函数的 consumer 本地 SRAM 里圈一段地址，给 FIFO 用 |
| `pto.import_reserved_buffer` | 在 producer 函数里引用 peer 函数那段地址 |

数据传输 op 上的常用属性：

- `id`：找哪条 pipe。
- `split`：切分模式，必须满足绑定 pipe 上 `nosplit` 的约束。
- `tpop_from_aic` / `tpop_from_aiv` 还可以带可选的 `valid_row` 和 `valid_col`，用于动态 valid shape 的 tile entry（详见 §2.7）。

### 2.2 取数据和释放 slot 是两步：tpop + tfree

`tpop` 只做"等数据就绪 + 把数据拿过来"，不发释放信号；释放交给 `tfree` 做。

为什么要分两步？consumer 拿到 tile 之后可能还要算一阵。如果 `tpop` 立刻把 slot 标成 free，producer 就可能在 consumer 还没读完时就覆盖它，A5 上零拷贝路径尤其危险——consumer 读的就是 producer 写的那块物理 SRAM。`tfree` 让 consumer 自己决定什么时候真的放回去。

注意：每个 `tpop_*` 都要配一个 `tfree_*`，并且要在环形 FIFO 转一圈回到同一个 slot 之前发出来，否则 producer 会卡死等 free。

### 2.3 两种 pipe entry：tile 还是 global

数据传输 op 都接收一个 pipe entry，entry 可以是两种形式：

| 类型 | MLIR 类型 | 数据怎么搬 | 适用场景 |
|---|---|---|---|
| tile entry | `!pto.tile_buf<...>` | 底层 TPUSH / TPOP 自动搬 | 当前主流。A2 / A3 通过 GM 中转，A5 零拷贝 |
| global entry | `!pto.tensor_view<...>` | 用户自己写 `pto.tstore` / `pto.tload` | 只走 A2 / A3。pipe 只负责同步和给 entry 算 GM 地址 |

global entry 是为了应对单个 entry 太大、需要 consumer 分多次读取子区域的场景。流程是：

- producer：`talloc_to_*` 拿到一个 GM slot 的 view → 自己 tstore 写进去 → `tpush_to_*` 通知 consumer。
- consumer：`tpop_from_*` 拿到 view → 自己 tload（也可以从 view 派生子 view 再 tload）→ 读完 `tfree_from_*`。

A5 上不能用 global entry，因为 FIFO 在 consumer 本地 SRAM 里，没有 GM 地址可以塞给 view。

### 2.4 A5 上的地址跨函数传递

A5 上 FIFO 在 consumer 本地 SRAM 里，但是 producer 也得知道这个地址才能 DMA 进去。C++ 语义里一个函数的局部符号在另一个函数看不到，得专门解决这个跨函数地址引用问题。

做法是：

1. consumer 函数写 `pto.reserve_buffer { name, size, location, auto }`，圈一段本地 SRAM。
2. producer 函数写 `pto.import_reserved_buffer { name, peer_func }`，引用 consumer 那段。
3. PTOAS 的 `pto-resolve-reserved-buffers` pass 把两边解析成同一个常量地址。

`reserve_buffer` 有两种用法：

- `auto = true`：地址由 `pto-plan-memory` 在本函数 local 地址空间剩余的空洞里挑一段。
- `auto = false` + 显式 `base`：跳过 plan memory，前端或更早阶段已经规划好了。

### 2.5 DIR_MASK = 3 (双向) 的 flag 占用

- 一条 `DIR_MASK = 3` 的前端 pipe 在 PTOAS 内部 lowering 成一条 DIR_BOTH 内部 pipe，不会拆成两条。
- 这条 pipe 同时承载 C2V 和 V2C，`slot_num = 4`。
- 占用两组逻辑 flag 对：`B/B+1` 给一个方向，`B+2/B+3` 给另一个方向，宽度等于两条单向 pipe。
- 一个函数里所有 pipe 占用的 flag 加起来不能超过 16。

### 2.6 tpop 上的 valid_row / valid_col（动态 valid shape）

`tpop_from_aic` 和 `tpop_from_aiv` 上可以挂一对可选的 `Index` 操作数 `valid_row` / `valid_col`：

- 仅对 tile entry 路径生效；global entry 路径忽略这两个操作数。
- 仅在 tile result 的 `v_row` / `v_col` 同时为 `?` 时合法；如果 tile 是静态 valid shape，给了 `valid_row` / `valid_col` 会被 verifier 拦下。
- 两个要么都给、要么都不给；只给一个会报错。
- lowering 时会在 popped tile 上插入一个 `pto.set_validshape`，运行时把 valid shape 设上去。

示例：

```mlir
%vr = ... : index
%vc = ... : index
%tile = pto.tpop_from_aic(%vr, %vc) {id = 0, split = 0}
  -> !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                   v_row=?, v_col=?, ...>
```

### 2.7 编译流程

```
前端 IR
  (aic/aiv_initialize_pipe + talloc_to_* / tpush_to_* / tpop_from_* / tfree_from_*)
        │
        ▼   lowering pass：绑定 pipe 方向，生成内部 IR
        │
PTOAS 内部 IR
  (initialize_l2l_pipe / initialize_l2g2l_pipe + talloc / tpush / tpop / tfree)
        │
        ▼   pto-infer-validate-pipe-init：
        │       - 推断 init op 上缺省的 nosplit
        │       - 校验该 pipe 上所有传输 op 的 split 与 nosplit 一致
        │       - 跨 peer 对齐两侧的 nosplit
        │
        ▼   pto-plan-memory：只在 auto = true 时跑，给 reserve_buffer 挑地址
        │
        ▼   pto-resolve-reserved-buffers：
        │       - 把 import_reserved_buffer 替换成常量地址
        │       - 把 peer 两侧的 flag_base 对齐
        │
        ▼   EmitC：透传到底层 TPipe / TALLOC / TPUSH / TPOP / TFREE
        │
pto-isa C++ 代码
```

---

## 3. 测试要点

### 3.1 建议覆盖的组合

| 维度 | 取值 |
|---|---|
| 平台 | A2 / A3、A5 |
| 方向 | C2V (1)、V2C (2)、双向 (3) |
| Entry 类型 | tile entry，global entry（只 A2 / A3） |
| split | TILE_NO_SPLIT、TILE_UP_DOWN、TILE_LEFT_RIGHT |
| nosplit | true（搭配 split=0）、false（搭配 split=1/2）、缺省（由 pass 推断） |
| AIV_IDX | 0、1 |
| 地址路径（A5 tile entry） | auto = true（走 PlanMemory）、auto = false（前端显式 base） |
| pipe 数量 | 单 pipe、同函数多 pipe（验证 flag 不重叠且总和不超过 16） |
| tpop 动态 valid shape | 静态 valid shape（无 valid_row/col）、动态（带 valid_row/col） |

### 3.2 应该被报错的非法 IR

下面这些情况 verifier 或 pass 应该报错，建议构造对应的负向用例。

前端层：

- `DIR_MASK` 取 0 或 ≥ 4
- `SLOT_SIZE <= 0`
- 同一函数内两条 init 用了相同 `id`
- 数据传输 op 的 `id` 找不到匹配的 init
- C2V 方向的 op 匹配到 `dir_mask = 2` 的 init
- `talloc_to_aiv` 写在 Vector kernel 里，或者 `talloc_to_aic` 写在 Cube kernel 里
- `reserve_buffer` 名字重复，或 `import_reserved_buffer` 的 `(name, peer_func)` 重复
- `reserve_buffer.size` 不等于 `SLOT_SIZE * SLOT_NUM`
- C2V consumer 的 `reserve_buffer.location` 不是 VEC
- V2C consumer 的 `reserve_buffer.location` 不是 MAT
- `auto = true` 但带了 `base`；`auto = false` 但没带 `base`
- global entry 的 `tpush` 在 IR 里没有支配它的 `talloc`
- global entry 路径下 `tfree` 没带 entry operand
- tile entry 路径下 `tfree` 带了 entry operand
- 同一次 transaction 里混用 tile entry 和 global entry
- global entry 绑到 A5 的 `initialize_l2l_pipe`（A5 没有 GM FIFO）
- 初始化 op 写 `nosplit = true`，但同 pipe 上的传输 op 用 `split = 1` 或 `2`
- 初始化 op 写 `nosplit = false`，但同 pipe 上的传输 op 用 `split = 0`
- 同 pipe 上的多个传输 op 用了不一致的 split（推断后 nosplit 自相矛盾）
- peer 两侧的 `nosplit` 显式值冲突
- `tpop_from_aic` / `tpop_from_aiv` 只给了 `valid_row` 或只给了 `valid_col`（缺一个）
- 给了 `valid_row` / `valid_col`，但 tile result 的 `v_row` / `v_col` 是静态值（不是 `?`）
- 把 `valid_row` / `valid_col` 用在 global entry 的 tpop 上（实现会忽略，但建议测试覆盖一下）

内部 IR 层：

- `slot_num` 不是 8 也不是 4
- `local_slot_num > slot_num`
- `dir_mask = 1` 的 pipe 被 V2C 方向 op 引用
- 进入 EmitC 时 `flag_base` 还没填好

资源 / pipeline 配置：

- 一个函数里 pipe 的 flag 加起来超过 16
- 同函数两条 pipe 的 flag 区间重叠
- peer 两侧显式 `flag_base` 不一致
- 启用 plan memory 的流程里出现 `auto = false`，或跳过 plan memory 的流程里出现 `auto = true`
- `import_reserved_buffer` 在 `peer_func` 里找不到同名的 `reserve_buffer`

### 3.3 功能 / 数值正确性

下面这些不会被 verifier 拦下，需要跑起来对结果。

- 环形回绕：迭代次数 ≥ SLOT_NUM + 1（单向至少 9 次、双向每方向至少 5 次），看 wrap 一圈以后 producer 还能等到 free、consumer 还能等到 ready。
- 背压：producer 快时应该卡在 `WAIT flag_free`；consumer 快时应该卡在 `WAIT flag_ready`。
- 顺序：consumer 拿到的顺序和 producer 写入的顺序严格一致。
- 延后 free：consumer 在 `tpop` 和 `tfree` 之间挂着不读 slot 一段时间，producer 不应该覆盖它（A5 零拷贝路径下特别重要）。
- 双向独立：双向场景下两个方向各自推进，一个方向阻塞不应该误锁另一个方向（除非应用层就有反向依赖）。
- AIV_IDX 路由：Cube push 给 AIV_IDX = 0 的数据，不能被 Vector1 拿到；反过来也是。
- 平台切换：同一份前端 IR 在 A2 / A3 和 A5 下生成的代码都能跑通，A5 零拷贝路径下 consumer 读到的数据和 producer 写入的一致。
- global entry 多次读：从同一个 `tpop` 拿到的 view 派生多个子 view 反复 tload，每次读出来的数据都一致。
- `split` sub-core offset：`TILE_UP_DOWN` / `TILE_LEFT_RIGHT` 下两个 sub-core 读到的 entry 子区域不重叠，并且拼起来正好是完整 entry。
- `nosplit` 模式（A5）：`nosplit = true` 下只用一个 Vector core 跑 pipe，跑通且数据正确。
- `nosplit` 模式（A2 / A3）：`nosplit = true` 下两个 Vector core 跑相同代码，pipe op 的相对顺序在两边保持一致；故意把两边顺序写错应该能复现卡死或数据错。
- 动态 valid shape：`tpop_from_*` 带上 `valid_row` / `valid_col`，consumer 拿到的 tile valid shape 应该是这两个运行时值，后续运算（比如 tcmp、tmov）按这个 valid shape 行事。

### 3.4 EmitC 输出可以顺便检查

做端到端 codegen 测试的话，可以顺便看一下 EmitC 出来的 pto-isa 长什么样：

- 初始化对应 `TPipe<flagBase, Direction::DIR_C2V | DIR_V2C | DIR_BOTH, ...>`，DIR_BOTH 时构造函数同时接收 C2V 和 V2C 两个 consumer buffer 地址。
- `pto.talloc` → `TALLOC<Pipe, GlobalData, Split>`
- `pto.tpush` → `TPUSH<Pipe, Tile|GlobalData, Split>`
- `pto.tpop` → `TPOP<Pipe, Tile|GlobalData, Split>`
- `pto.tfree` → `TFREE<Pipe, Split>` 或 `TFREE<Pipe, GlobalData, Split>`
- 进入 EmitC 前，`reserve_buffer` / `import_reserved_buffer` 和所有前端 op 都应该已经消除。

---

## 4. 完整示例：A2 / A3 上的 C2V global entry

```mlir
func.func @cube_kernel(%gm_slot_buffer : !pto.ptr<f32>,
                       %src : !pto.tile_buf<...>)
    attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %gm_slots = pto.make_tensor_view %gm_slot_buffer,
    shape = [%c16, %c16], strides = [%c16, %c1]
    : !pto.tensor_view<16x16xf32>

  pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
    (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

  %entry = pto.talloc_to_aiv {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf32>
  %sub = pto.partition_view %entry,
    offsets = [%c0, %c0], sizes = [%c16, %c16]
    : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
  pto.tstore ins(%src : !pto.tile_buf<...>)
             outs(%sub : !pto.partition_tensor_view<16x16xf32>)
  pto.tpush_to_aiv(%entry : !pto.tensor_view<16x16xf32>) {id = 0, split = 0}
  func.return
}

func.func @vector_kernel(%gm_slot_buffer : !pto.ptr<f32>,
                         %dst : !pto.tile_buf<...>)
    attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %gm_slots = pto.make_tensor_view %gm_slot_buffer,
    shape = [%c16, %c16], strides = [%c16, %c1]
    : !pto.tensor_view<16x16xf32>

  pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
    (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

  %entry = pto.tpop_from_aic {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf32>
  %sub = pto.partition_view %entry,
    offsets = [%c0, %c0], sizes = [%c16, %c16]
    : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
  pto.tload ins(%sub : !pto.partition_tensor_view<16x16xf32>)
            outs(%dst : !pto.tile_buf<...>)
  pto.tfree_from_aic(%entry : !pto.tensor_view<16x16xf32>) {id = 0, split = 0}
  func.return
}
```

几个要点：

- 两侧用同一个 `id = 0` 把数据传输 op 和初始化 op 绑起来。
- 这是 global-only GM FIFO 路径：init 只传 `gm_slot_tensor`，不传 `gm_slot_buffer` / `local_slot_num` / `c2v_consumer_buf` / `v2c_consumer_buf`。
- 这条路径下不需要 `reserve_buffer` 和 `import_reserved_buffer`。
- `pto.tstore` 和 `pto.tload` 都是用户显式写的，PTOAS 不会自动插入。
- `tfree_from_aic` 必须带 entry operand，并且这个 entry 是 `tpop_from_aic` 返回的那个。
