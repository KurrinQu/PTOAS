# Tile Fusion 详细设计文档大纲

## 1. 引言

### 1.1 背景与动机

- **PTO 指令执行模型**：分析以 UB 上的 Tile 为媒介的存取开销。
  - **存取开销**：在 Davinci 架构中，PTO 指令（如 `pto.add`）通常以 UB 上的数据块（Tile）为输入输出。当多条 PTO 指令连续执行时，中间结果必须写回 UB 再重新读入向量寄存器，产生冗余的存储带宽消耗。
  - **控制开销**：每条 PTO 指令在底层都会展开为针对 Tile 形状的硬件循环。多条 PTO 指令意味着多组循环初始化、分支预测和指令发射开销。
- **示例分析：`D = Relu(Add(A, B))`**
  - **非融合模式**：
    1. **Loop 1 (Add)**: 从 UB 读取 Tile A, B -> 加法运算 -> 结果 Tile C 写回 UB。
    2. **Loop 2 (Relu)**: 从 UB 读取 Tile C -> Relu 运算 -> 结果 Tile D 写回 UB。
    - _瓶颈_：Tile C 的一次写和一次读完全冗余；存在两组循环控制逻辑。
  - **融合模式**：
    1. **Single Loop**: 从 UB 读取 Tile A, B -> 加法运算 -> **寄存器直接传递** -> Relu 运算 -> 结果 Tile D 写回 UB。
    - _收益_：消除了 Tile C 的 UB 访存；两组循环合并为一组，降低了 50% 的循环管理开销。

- **图示：存取与控制流对比**

```text
      [非融合模式]                           [融合模式]
      (Loop 1: Add)                         (Single Loop)
    ┌───────────────┐                     ┌───────────────┐
    │ Read A, B     │                     │ Read A, B     │
    │      │        │                     │      │        │
    │ Compute Add   │                     │      │        │
    │      │        │                     │      │ (Reg)  │
    │ Write C to UB │──┐ (UB Latency)     │ Compute Relu  │
    └───────────────┘  │                  │      │        │
    (Loop 2: Relu)     │                  │ Write D to UB │
    ┌───────────────┐  │                  └───────────────┘
    │ Read C fr UB  │◀─┘
    │      │        │               >> 消除中间写回/读取
    │ Compute Relu  │               >> 合并循环控制逻辑
    │      │        │
    │ Write D to UB │
    └───────────────┘
```

- **现有瓶颈总结**：多条 PTO 指令间频繁的 UB 读写导致带宽受限、循环控制指令占比过高、以及硬件计算单元在等待存取时出现空闲。

### 1.2 设计目标与范围

- **融合范围界定**：本特性首期聚焦于 `docs/PTO_IR_manual.md` 中所有硬件映射为 **Vector Pipeline (`PIPE_V`)** 的 PTO 指令。
  - **支持的 OP 范围**：
    - **向量算术运算**：包括逐点运算如 `pto.tadd`, `pto.tmul`, `pto.tdiv`, `pto.tmax`, `pto.tprelu` 等。
    - **向量标量运算**：如 `pto.tadds`, `pto.tmuls`, `pto.tmaxs` 等。
    - **向量规约与广播**：如 `pto.trowsum`, `pto.tcolmax`, `pto.tbcast` 等。
    - **向量位运算**：如 `pto.tand`, `pto.tor`, `pto.txor` 等。
    - **局部数据移动**：如 `pto.tmov` (ACC <-> VEC 域移动), `pto.ttrans` (转置) 等。
  - **不支持范围**：不包含 Matrix Pipeline (`PIPE_M`) 的矩阵乘法指令或 DMA Pipeline (`PIPE_MTE`) 的搬运指令内部逻辑融合，但这些指令可作为融合链路的起点或终点。
- **支持的融合模式与准则**：
  并非任意两条 `PIPE_V` 指令都可融合，必须满足以下核心准则：
  1. **迭代空间一致性**：
     - **定义**：参与融合的 OP 必须在相同的逻辑迭代区间内运行。对于 PTO Tile 而言，这意味着其逻辑计算形状（Valid Shape，即 `v_row` 和 `v_col`）必须能够对齐到同一个循环空间。
     - **物理 vs 逻辑解耦**：物理形状代表内存分配大小，可以不一致；但逻辑形状代表计算范围，必须匹配。
     - **支持场景示例**：
       - **场景 A (完全匹配)**：OP1 和 OP2 的逻辑形状均为 `16x128`。融合后生成一个 `16 * 128` 的单一硬件循环。
       - **场景 B (物理布局差异)**：OP1 的 Tile 物理大小为 `32x128`，但有效区域 `v_row=16, v_col=128`；OP2 的 Tile 物理大小为 `16x128`，有效区域亦为 `16x128`。由于逻辑迭代空间一致，可以融合。
       - **场景 C (异构输出形状但同域)**：`OP1: C = Add(A, B)` (输出 Tile 16x128) 与 `OP2: d = RowSum(C)` (输出 Vector 16x1)。
     - **不支持场景示例**：
       - **场景 D (逻辑形状冲突)**：OP1 逻辑形状为 `16x128`，OP2 逻辑形状为 `8x128`。由于循环上限不一致，直接融合会导致迭代次数错误，首期暂不支持此类非对齐融合。
       - **场景 E (动态形状不可证)**：OP1 的 `v_row` 为动态值 `%M1`，OP2 的 `v_row` 为 `%M2`。若编译器无法静态证明 `%M1 == %M2`（例如通过符号分析证明其来自同一个 `pto.get_tensor_view_dim`），则为了安全起见不予融合。
  2. **数据依赖与通用映射准则**：
     - **逐点 OP 的融合普适性**：逐点 OP（如 `Add`, `Mul`, `Relu`）因其不改变数据索引映射关系，是构建融合链的核心灵活性单元。
     - **准则与约束条件**：
       - **弱依赖准则**：融合不强制要求 OP 间存在数据流依赖。处于同一迭代空间的并行独立 OP 只要满足资源约束，即可通过合并循环逻辑实现“控制流融合”。
       - **深度优化准则**：当 OP 间存在生产者-消费者关系且满足点对点映射（1:1）时，触发“存取消除融合”，利用向量寄存器直传彻底省去中间 Tile 的 UB 读写。
       - **规约适配准则**：在 A5 硬件支持下，逐点 OP 与规约（Reduce）OP 的组合同样遵循深度融合准则，支持在寄存器内完成即时累加并消除访存。
     - **场景示例**：
       - **场景 F (并行无依赖融合)**：`OP1: C = Add(A, B)` 和 `OP2: F = Mul(D, E)`。
         - _分析_：两者虽无依赖，但合并为单循环可显著减少循环管理指令，提升计算单元填充率。
       - **场景 G (逐点深度融合 - 1:1 映射)**：`OP1: C = Add(A, B)`，`OP2: D = Relu(C)`。
         - _分析_：最典型的 1:1 映射。`C` 直接在寄存器中由 `Relu` 消耗，完全省去中间 Tile 的物理地址分配。
       - **场景 H (跨模式融合 - 逐点 + 规约)**：`OP1: C = Add(A, B)`，`OP2: d = RowSum(C)`。
         - _分析_：在 A5 上，`RowSum` 指令可以在完成加法计算的同时，在向量寄存器内部进行跨列累加。融合后不仅合并了循环，还消除了大尺寸 Tile `C` 在 UB 上的存储和加载开销。
  3. **内存布局一致性与弹性适配**：
     - **标准准则（布局一致的具体含义）**：参与融合的输入/输出 Tile 在元数据定义和硬件存取模式上必须完全兼容。具体包括：
       - **元数据对齐**：Tile 的基础布局（`blayout`，如行优先/列优先）、二级布局（`slayout`）、分形大小（`fractal`）以及数据类型（`dtype`）必须完全一致。
       - **内存映射一致**：在硬件层面，这意味着当数据从 UB 加载到向量寄存器时，Tile 中逻辑索引为 `(i, j)` 的元素在寄存器序列中的物理位置必须相同。
       - **冲突后果**：若布局不一致（例如 OP1 是行优先输出，OP2 要求列优先输入），即使迭代空间相同，它们在寄存器中的元素排列顺序也是“错位”的。因此，标准准则要求物理布局必须对齐，以支持零开销的寄存器级传参。
     - **特殊场景适配**：
       - **虚拟布局变换**：若中间 OP 仅为布局转换指令（如 `pto.ttrans` 转置），可通过修改计算循环的索引映射逻辑，将布局转换开销“消除”在向量寄存器的存取步长或偏置中。
       - **硬件重排加速**：利用 A5 硬件在向量寄存器内部的混叠或排列指令，在计算过程中实时调整数据顺序，从而支持不同布局 OP 的融合。
     - **场景示例**：
       - **场景 I (转置消除融合)**：`OP1: B = Transpose(A)` (使用 `pto.ttrans`)，`OP2: C = Relu(B)`。
         - _分析_：若单独执行，`Transpose` 需在 UB 进行物理搬运。通过融合，编译器可生成一个“按列遍历”的 `Relu` 计算核心直接处理 Tile `A`。逻辑上中间变量 `B` 消失了，且彻底消除了物理转置带来的冗余访存开销。
  4. **寄存器与硬件参数预算约束**：
     - **物理寄存器限制**：Davinci A5 拥有 32 个向量寄存器（$V_0$ - $V_{31}$）。融合链中的所有活跃变量（包括输入、中间结果、临时变量）总数不得超过此阈值，否则会产生寄存器溢出（Spill），导致数据被迫写回 UB/L1，从而严重拖慢性能。
     - **VF 参数限制**：向量函数（Vector Function, VF）硬件循环的参数列表（包括 Tile 物理地址、步长 Stride、形状 Shape 等元数据）总数上限为 32。如果融合后的单一大循环涉及过多的独立 Tile 对象，将导致 VF 调用参数超限。
     - **场景示例**：
       - **场景 J (参数与寄存器双重超限)**：尝试将一个涉及 18 个不同输入项的复杂融合链（如 18 个输入 Tile 连加）进行融合。
         - _分析_：
           1. **参数溢出**：18 个输入 Tile 地址 + 1 个输出 Tile 地址 + 对应 Tile 的多个 Stride 元数据，其总参数数量极易触碰 32 个 VF 参数的硬件天花板。
           2. **寄存器溢出**：若该计算链路由于循环展开（Unroll）导致有超过 32 个 128-bit 向量值同时处于活跃状态，硬件将无法在不写回内存的情况下完成寄存器级传参。
         - _决策_：此时编译器必须在中间点强制截断（Split）融合链，将中间结果写回 UB 并在两个独立的 VF 循环中执行。
  5. **典型支持模式**：
     - **线性链路**：`A -> B -> C` 的连续逐点运算。
     - **并行独立融合**：将处于同一迭代空间、且无直接数据流依赖的多个 OP 组合进同一个硬件循环。虽然不消除访存，但降低控制开销。
     - **广播融合**：`pto.tbcast` 后紧跟逐点运算。
     - **末端规约融合**：一系列逐点运算后紧跟 `pto.trowsum` 或 `pto.tcolmax`。
     - **规约 + 逐点**：规约指令的输出向量直接作为后续逐点指令的输入
     - **逐点 + 广播**：逐点运算的结果 Tile 直接作为广播指令的输入，在寄存器层面完成扩展
     - **规约 + 广播**：常见的 Softmax 结构优化。规约得到的行/列极值直接通过广播扩展回 Tile 形状进行后续计算。

### 1.3 核心思想

- **技术核心**：将多条 PTO 指令的循环体在 MLIR 层面进行融合，通过向量寄存器直接传递中间数据，从而实现以下核心优化目标：
  - **减少 UB 访存**：通过向量寄存器级的数据流水化替代物理 Tile 的中间写回与读取，消除冗余带宽消耗。
  - **降低控制开销**：将多组硬件循环合并为单一循环结构，减少循环初始化、分支预测及指令发射周期。
  - **提升计算能效**：优化指令重排与 Unroll，使向量计算单元尽可能保持在满载状态。
  - **透明性**：为上层编译器框架提供明确的融合边界，开发者可预期哪些指令组合能够获得寄存器级加速。

## 2. 硬件架构与约束

### 2.1 Davinci A5 存储层次与访存成本

- **存储体系结构**：
  - **Global Memory (GM)**：百 GB/s 级带宽，高访存延迟（数百周期）。
  - **Unified Buffer (UB)**：TB/s 级带宽，中等访存延迟。PTO 指令默认以此为输入输出。
  - **向量寄存器 (Vector Register)**：指令级单时钟周期访问。
- **融合驱动力**：PTO 指令间若通过 UB 传递数据，会受限于 UB 的读写带宽（Bandwidth Bound）；通过寄存器直传可将数据流提升至寄存器级带宽，消除物理存取延迟。

### 2.2 向量计算管道与执行机制

- **双发射流水线**：Davinci 支持两组向量计算流水线（Pipe V），通过合理的指令重排（Scheduling）可隐藏计算延迟。
- **硬件循环 (Vector Function)**：硬件内置循环控制器，初始化开销相对较大。融合后的“单大循环”能有效摊薄循环头（Prologue）和循环尾（Epilogue）的指令周期。
- **向量掩码 (VMSK)**：用于处理非对齐边界。融合时需确保不同 OP 的掩码逻辑在同一个循环迭代内是可组合的。

### 2.3 物理对齐与 Tile 布局约束

- **32B/512B 对齐要求**：Davinci 架构对 UB 地址及 Tile 宽度有严格的对齐要求。
- **Padding 填充**：当逻辑形状（Valid Shape）小于物理形状时，硬件会在物理 Tile 边缘填充无效数据。融合时必须严格控制计算边界，防止对无效填充区进行错误累加。
- **分形布局机制**：针对矩阵/分形操作的特殊布局。向量运算在处理此类布局时需要额外的 Shuffle 或 Stride 计算。

## 3. 融合的难点与挑战

### 3.1 寄存器压力

- **Spill 风险**：融合多个 OP 会显著增加活跃变量的数量。如果所需向量寄存器超过硬件上限，会产生寄存器溢出，导致数据写回 UB 或 L1，从而抵消融合带来的访存收益。
- **生命周期重叠**：融合后的长循环会延长中间结果的生命周期，增加寄存器分配算法的复杂度。

### 3.2 内存布局与对齐

- **布局不匹配**：例如 Producer OP 输出 Row-major 数据，但 Consumer OP 要求 Col-major 输入。融合此类 OP 可能需要插入昂贵的转置或重排指令，抵消访存收益。
- **非逐点操作**：涉及 Reduction 或 Data Movement 的 OP 融合涉及复杂的索引变换和同步逻辑。

### 3.3 循环控制与同步

- **循环结构差异**：具有不同遍历范围或步长的 OP 融合时，需要精细的循环剥离或填充技术。
- **细粒度同步**：在某些流水线架构中，融合后的长指令序列可能导致硬件流水线死锁或违反数据依赖顺序。

### 3.4 成本模型与边界决策

- **贪心策略的局限性**：过度融合可能导致代码膨胀和指令 Cache 缺失。
- **自适应决策**：如何在不同 Tile Shape 和硬件配置下自动决定融合边界，是一个非确定性多项式难题。

## 4. IR 表示与现有流程

### 4.1 PTO Dialect 现状

- **核心指令**：`pto.alloc_tile`, `pto.load_tile`, `pto.store_tile` 等。
- **OP 级表示**：现有的 OP 级表示及其降低路径。

### 4.2 OP 融合在流程中的位置

- **当前生效条件**：仅在 `tools/ptoas/ptoas.cpp` 的 A5/A5VM backend mainline 中启用，要求同时满足：
  - `--pto-arch=a5`
  - backend 选择 A5VM
  - 显式传入 `--enable-op-fusion`
- **当前主线位置**：
  1. 先在 `tile_buf` 语义下执行 `FusionPlan -> OpScheduling -> PTOFusionRegionGen`。
  2. 再进入 shared pre-backend normalization：`LoweringSyncToPipe -> InferPTOLayout -> PTOViewToMemref -> PlanMemory -> (可选) PTOInsertSync -> CSE`。
  3. 然后进入 A5VM backend 主线：`PTOA5VMVersionSelection -> PTOToA5VM -> PTOA5VMIfCanonicalize -> PTOLowLevelLoopFusion -> Canonicalizer -> CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion -> CSE`。
- **非目标路径**：
  - EmitC backend 会忽略 `--enable-op-fusion`。
  - 当前 A5 backend 主线不再串接旧的 OP-Lib instantiate / inline 路径；tile fusion 的 backend lowering 基线是 `PTOToA5VM`，不是历史上的 LibCall/CCE 主线。

## 5. 详细技术设计

### 5.0 融合转换流水线 (Fusion Pipeline Overview)

```text
    [Level-2 PTO IR / tile_buf world]
               │
               ▼
    5.1 预融合分析 (PreFusionAnalysis, analysis-only)
               │
               ▼
    5.2 迭代域证明 (当前内嵌于预分析, 无独立 ShapeInferencePass)
               │
               ▼
    5.3 融合分组规划 (FusionPlan)
               │
               ▼
    5.4 组内物理聚拢 (OpScheduling)
               │
               ▼
    5.5 区域封装 (PTOFusionRegionGen)
               │
               ▼
    5.6 Shared pre-backend normalization
        (Sync lowering / layout infer / view->memref /
         plan memory / optional insert-sync / CSE)
               │
               ▼
    5.7 A5VM 版本选择与 lowering
        (PTOA5VMVersionSelection -> PTOToA5VM)
               │
               ▼
    5.8 A5VM 前置结构清理 (PTOA5VMIfCanonicalize)
               │
               ▼
    5.9 低层循环融合 (PTOLowLevelLoopFusion)
               │
               ▼
    5.10 后融合规范化与谓词消除
         (Canonicalizer -> CSE -> PTOFusionPredicateElision)
               │
               ▼
    5.11 融合区域内 Load/Store 消除
         (PTOFusionLoadStoreElision)
               │
               ▼
    5.12 展平区域包装 (PTOFlattenFusionRegion)
               │
               ▼
    5.13 A5VM backend 发射
         (A5VM text / LLVM emission)
```

### 5.1 全局依赖分析 (PreFusionAnalysis，analysis-only)

**设计动机 (Motivation)**
为 `FusionPlan` 提供 block-local 的可复用分析结果，而不是直接改写 IR。当前实现把“哪些 op 能参与 tile fusion”“哪些值跨越 local/hard boundary 逃逸”“哪些 op 处于同一迭代域”都前置到统一分析里，供规划、封装和后续 region-local 清理复用。

**流水线位置 (Pipeline Position)**

- **Pre-conditions**：IR 仍在 `tile_buf` 语义世界，尚未经过 `PTOViewToMemref`。
- **Post-conditions**：生成 `PreFusionAnalysis` 结果对象；默认主线不单独插入这个 pass，而是由 `FusionPlanPass` 通过 `getAnalysis<pto::PreFusionAnalysis>()` 直接消费。

**输入/输出规格 (I/O Specification)**

- **Input**: `func::FuncOp` 内的 PTO tile-level IR。
- **Output**: 每个 basic block 的 `FusionBlockAnalysis`，主要包含：
  - `computeNodes`：可参与预融合建模的 compute node。
  - `edges`：producer/consumer 依赖边。
  - `valueLiveness` / `writeInstances`：value 和写实例的生命周期与逃逸类别。
  - `iterationDomainClasses`：按 `(v_row, v_col)` 证明结果划分的迭代域类。

**核心逻辑与约束 (Logic & Constraints)**

- **Op 语义分类**：`FusionOpSemantics` 先把 op 分成 `Compute`、`LocalBoundary`、`HardBoundary`。
  - `treshape` 当前被视为 `LocalBoundary`，会阻断穿越它的局部规划与调度。
  - 非 `OpLibOpInterface`、带 region/call/未知副作用的 op，一律按 `HardBoundary` 处理。
- **当前可识别的 compute family**：
  - `Elementwise`：`tadd/tsub/tmul/tdiv/tmax/tmin` 及对应 scalar 版本，`texp`
  - `ScalarExpand`：`texpands`
  - `RowBroadcastBinary`：`trowexpandmul`、`trowexpanddiv`
  - `ReduceRow/ReduceCol`：`trowsum/trowmax/trowmin`、`tcolsum/tcolmax/tcolmin`
- **依赖与生命周期建模**：
  - 通过 SSA use-def 和 DPS 输出归一化收集 tile 输入/输出。
  - 记录 block 内 consumer、local boundary user、hard boundary user、块外逃逸信息。
  - 写实例会进一步区分 `Internal`、`LocalBoundaryExternal`、`HardExternal`，供 region output/frontier 计算使用。
- **分析范围**：当前严格限制在 basic block 内，不做跨块、跨 CFG 边的全局规划。

**不变性与副作用 (Invariants & Side Effects)**

- 纯分析，不修改 IR。
- 可通过 `pto-pre-fusion-analysis` / `pto-print-pre-fusion-analysis` 做调试或 lit 检查，但它们不是默认 backend 主线的一部分。

### 5.2 迭代域证明 (当前无独立 ShapeInferencePass 主线)

**设计动机 (Motivation)**
当前代码里“形状推导”还没有落成独立 pass。真正被主线依赖的是 `FusionAnalysis.cpp` 中对迭代域的一次保守证明：只有当参与计算的 anchor tile 能在编译期证明为一致的 rank-2 有效形状时，规划阶段才会继续融合。

**流水线位置 (Pipeline Position)**

- **Pre-conditions**：依赖 5.1 中的 op 语义和 tile 类型/`pto.bind_tile` 元数据。
- **Post-conditions**：不会改写 IR；只会给分析结果里的 `IterationDomainInfo` 标注 `Proven/Unproven` 及失败原因。

**输入/输出规格 (I/O Specification)**

- **Input**: compute op 的 tile 输入/输出以及其 valid shape 信息。
- **Output**: `IterationDomainInfo { vRow, vCol, proof, unprovenReason }`。

**核心逻辑与约束 (Logic & Constraints)**

- **当前证明来源**：
  - 优先读取 `TileBufType` 的 `validShape`。
  - 若值来自 `pto.bind_tile`，会用其 `validRow/validCol` 常量覆盖类型上的静态形状。
  - `Elementwise` 会聚合输入和输出；`ScalarExpand/RowBroadcastBinary` 用输出域；`ReduceRow/ReduceCol` 用输入域。
- **当前失败情形**：
  - 任一关键维度是动态值。
  - 同一组 anchor 的 `(v_row, v_col)` 不一致。
  - 缺失可恢复的 tile domain 信息。
- **现实边界**：代码注释已经明确，当前实现“不尝试证明动态符号等价”。因此带动态 `v_row/v_col` 的链路会保持 `Unproven`，并在 5.3 的规划阶段被保守拒绝。

**不变性与副作用 (Invariants & Side Effects)**

- 不生成新 attribute，不重写类型，不引入新的 shape solver。

### 5.3 融合策略与成本模型 (FusionPlanPass)

**设计动机 (Motivation)**
根据 5.1/5.2 的分析结果，给 block 内 op 打上稳定的组元数据，形成后续调度和区域封装的唯一输入契约。当前实现强调“保守可用”，而不是开放式策略插件框架。

**流水线位置 (Pipeline Position)**

- **Pre-conditions**：`PreFusionAnalysis` 有效，且 op 仍然处于 tile-level PTO IR。
- **Post-conditions**：被接受的组成员获得：
  - `pto.fusion.group_id`
  - `pto.fusion.order`

**输入/输出规格 (I/O Specification)**

- **Input**: `FusionBlockAnalysis`。
- **Output**: 带规划 metadata 的原始 PTO IR，且只有组大小 `>= 2` 的 group 才会真正落盘到 IR。

**核心逻辑与约束 (Logic & Constraints)**

- **当前实际策略**：
  - `ConservativeDAGGreedyStrategyEngine`
  - `ConservativeDAGGreedyCostModel`
- **当前可规划 op 子集**：比 5.1 的“可分析 compute family”更窄，只接受：
  - `tadd/tsub/tmul/tdiv/tmax/tmin`
  - `tadds/tsubs/tmuls/tdivs/tmaxs/tmins`
  - `texp`
  - `texpands`
  - `trowexpandmul`
  - `trowexpanddiv`
- **seed 条件**：
  - op 必须属于上述可规划集合。
  - 所在迭代域类必须是 `Proven`。
- **append 条件**：
  - candidate 与当前组首成员属于同一 `iterationDomainClass`。
  - candidate 与当前组至少存在一条直接数据流连接。
  - 成本模型评分 `dependencyBenefit + loopMergeBenefit - liveTilePenalty - vfParameterPenalty > 0`。
- **当前成本模型参数**：
  - `dependencyBenefit = 4 * connectionCount`
  - `loopMergeBenefit = 4`
  - 当 `liveTileCount > 10` 时开始罚分
  - 当 `vfParameterCount > 12` 时开始罚分
- **结果排序**：
  - 组内顺序按 `blockOrder/id` 稳定排序。
  - `group_id` 也按组首成员的 block 顺序稳定分配。

**不变性与副作用 (Invariants & Side Effects)**

- 只打 metadata，不移动 op。
- 当前没有把策略接口暴露成可插拔配置；文档中的“ML/AI 决策接口”仍然属于未来方向，不是现状。

### 5.4 OP 调度优化 (OpSchedulingPass)

**设计动机 (Motivation)**
把 5.3 已规划好的 group 压缩成 block 内连续 span，为 `PTOFusionRegionGen` 提供“一组对应一个连续区间”的结构前提。

**流水线位置 (Pipeline Position)**

- **Pre-conditions**：op 已经带有完整的 `pto.fusion.group_id/order`。
- **Post-conditions**：每个 group 在 block 中形成一个连续 span，group membership 不变。

**输入/输出规格 (I/O Specification)**

- **Input**: 带规划 metadata 的 PTO IR。
- **Output**: 物理顺序重排后的 PTO IR。

**核心逻辑与约束 (Logic & Constraints)**

- **barrier 分类**：
  - `Movable`：普通可移动 compute op。
  - `LocalBoundary`：例如 `treshape`，允许在无 tile 依赖冲突时跨越。
  - `HardBoundary`：call、region op、未知副作用 op、不可安全移动的边界。
- **调度策略**：
  - 先按 `group_id` 收集成员，再按 `pto.fusion.order` 排序。
  - 对组内后续成员，优先尝试把成员移动到当前 `placement` 之后。
  - 若成员不能前移，则在不违反 later-crossing 约束时，反向尝试把 `placement` 向后推。
- **关键合法性检查**：
  - 不能跨越 operand 的定义点。
  - 不能把 producer 挪到某个 consumer 之后。
  - 不能越过 hard boundary。
  - 穿越 local boundary 时，要确认双方不共享 tile input/output 依赖。

**不变性与副作用 (Invariants & Side Effects)**

- 会改写 block 内 op 顺序。
- 不改 group metadata，不改 CFG。

### 5.5 融合区域封装 (PTOFusionRegionGenPass)

**设计动机 (Motivation)**
把连续 span 封装成显式 `pto.fusion_region`，为 A5VM lowering 后的 region-local 循环融合、谓词清理和 load/store 消除提供容器边界。

**流水线位置 (Pipeline Position)**

- **Pre-conditions**：同一 `group_id` 在 block 中已经是单一连续 span。
- **Post-conditions**：每个 span 被包成一个 `pto.fusion_region`，并用 `pto.yield` 显式声明对外可见 frontier。

**输入/输出规格 (I/O Specification)**

- **Input**: 已调度好的 block-local group span。
- **Output**: `pto.fusion_region` 包装后的 PTO IR。
  - region 不显式建模输入 block argument。
  - region body 直接隐式捕获父作用域 SSA。
  - `pto.yield` 只返回真正对 region 外可见的 value。

**核心逻辑与约束 (Logic & Constraints)**

- **span 识别约束**：
  - 同一 `group_id` 在一个 block 里必须只出现一个连续 span，否则直接报错。
  - group 内 `pto.fusion.order` 必须严格递增。
- **frontier 计算**：
  - `PTOFusionRegionGen` 会结合 use-def 和 `PreFusionAnalysis` 的写实例逃逸信息，找出必须在 region 结果列表中保留的 escaping value。
  - 若某个值只在 region 内使用，或者虽然有外用但不可被 region result 合法替换，则不会盲目外提。
- **结构约束**：
  - 一个 group 对应一个 region。
  - 不额外生成 region 输入 operands。
  - 允许空 result / 空 `pto.yield`。

**不变性与副作用 (Invariants & Side Effects)**

- 会引入嵌套 region，显著改变 IR 结构。
- 这是当前主线里真正把“逻辑 fusion group”转成“后续 backend 可识别容器”的分界点。

### 5.6 Shared pre-backend normalization

**设计动机 (Motivation)**
tile fusion 分组和 region 封装完成后，主线并不会直接进入 A5VM emission，而是先执行一段与 A5 backend 共享的规范化流程，把同步、layout、view 和内存规划统一到 backend lowering 可消费的形态。

**流水线位置 (Pipeline Position)**

- **Pre-conditions**：`pto.fusion_region` 已经建立，但 body 仍然是 PTO tile op。
- **Post-conditions**：IR 已经过同步降级、layout 推断、view-to-memref 和内存规划，仍保留 `pto.fusion_region` 包装以便 backend-side region-local 优化继续工作。

**核心逻辑与约束 (Logic & Constraints)**

- 固定顺序如下：
  1. `PTOLoweringSyncToPipe`
  2. `InferPTOLayout`（除非显式禁用）
  3. `PTOViewToMemref`
  4. `PlanMemory`（`level3` 以外）
  5. `PTOInsertSync`（仅用户显式开启）
  6. `CSE`
- 关键现实点：
  - 低层循环融合不再发生在 “tile-level + LibCall inline” 之后，而是发生在这段 shared normalization 和 `PTOToA5VM` 之后。
  - 这一步仍然不会 flatten `pto.fusion_region`。

### 5.7 A5VM 版本选择与 lowering (PTOA5VMVersionSelection -> PTOToA5VM)

**设计动机 (Motivation)**
把 tile-level PTO op 变成 A5VM backend 可发射的低层 `scf.for + a5vm.*` 结构，同时为 fusion-region 内外选择不同的 lowering 变体。

**流水线位置 (Pipeline Position)**

- **Pre-conditions**：已完成 shared pre-backend normalization。
- **Post-conditions**：region 内 PTO op 被原位改写为 A5VM 低层结构；非融合 PTO op 也会在父 block 中被正常 lower。

**核心逻辑与约束 (Logic & Constraints)**

- **PTOA5VMVersionSelection**：
  - 会遍历所有 A5VM candidate PTO op。
  - 若 op 位于 `pto.fusion_region` 内，则打上 `pto.a5vm_lowering_choice = no-post-update`。
  - 若 op 位于普通父 block，则选择 `post-update`。
  - 当前 loop shape 固定为 `TwoD`。
- **PTOToA5VM**：
  - 会把 PTO tile op lowering 成 A5VM backend op。
  - 对已经封装好的 `pto.fusion_region`，采用“region 内原位改写，wrapper 暂时保留”的策略。
  - 不会为了 residual 非融合 op 人工再创建新的 `pto.fusion_region`。

### 5.8 A5VM 前置结构清理 (PTOA5VMIfCanonicalize)

**设计动机 (Motivation)**
`PTOToA5VM` 之后可能残留一些 `scf.if` 结构。当前主线在低层循环融合前，只做一轮针对 `scf.if` 的局部 canonicalization，避免全局 canonicalizer 过早改写 loop header，影响低层循环融合的结构匹配。

**流水线位置 (Pipeline Position)**

- **Pre-conditions**：输入是 A5VM post-lowering IR。
- **Post-conditions**：仅清理现有 `scf.if`，不主动跑全局 Canonicalizer。

**核心逻辑与约束 (Logic & Constraints)**

- 只应用 `scf::IfOp` 自带 canonicalization patterns。
- `GreedyRewriteConfig` 的 scope 限制在函数体内，目标是删除常量条件和冗余 if 包装，不破坏后续 loop-fusion 需要的 `scf.for` 头结构。

### 5.9 低层循环合并优化 (PTOLowLevelLoopFusion)

**设计动机 (Motivation)**
tile fusion 的真正“循环融合”现在发生在 A5VM lowering 之后。该 pass 不再操作历史上的 `pto.simd.vec_scope` / `vector.masked_*` bridge IR，而是直接在 `pto.fusion_region` 内处理低层 `scf.for + a5vm.*` 阶段。

**流水线位置 (Pipeline Position)**

- **Pre-conditions**：输入契约是保留在 `pto.fusion_region` 内的 A5VM post-lowering loop nest。
- **Post-conditions**：可融合的相邻 loop stage 被聚合成共享 loop-header 的单一 carrier loop。

**核心逻辑与约束 (Logic & Constraints)**

- **stage 识别方式**：
  - 每个 stage 由若干 setup op、若干层同构 `scf.for`、以及叶子 `a5vm.*` op 组成。
  - 只会尝试融合彼此相邻、loop header 等价、且中间 prelude/setup 可安全重排的 stage。
- **合法性条件**：
  - `sameForHeader(lhs, rhs)`：上下界、步长和 loop attrs 完全等价。
  - prelude/setup 必须是 side-effect-free 或可分析的 memory prelude。
  - 跨 stage 移动 prelude 时，不能与前一 stage 的内存根产生潜在 alias 冲突。
- **现实边界**：
  - 这是一个非常保守的 matcher，不做激进 loop normalization。
  - 只融合“相邻 stage”，不做跨区域或跨复杂控制流的全局循环拼接。

### 5.10 后融合规范化与谓词消除 (Canonicalizer -> CSE -> PTOFusionPredicateElision)

**设计动机 (Motivation)**
低层循环融合结束后，主线会先做一次常规 `Canonicalizer + CSE`，再专门清理 fusion-region 内部冗余的 A5VM 谓词物化，减少后续 load/store 消除阶段看到的噪声。

**流水线位置 (Pipeline Position)**

- **Pre-conditions**：`PTOLowLevelLoopFusion` 已完成。
- **Post-conditions**：重复的 `a5vm.plt_*` 计算被压缩，后续访存消除看到的 A5VM loop body 更干净。

**核心逻辑与约束 (Logic & Constraints)**

- `PTOFusionPredicateElision` 当前聚焦 `a5vm::PltB8/B16/B32Op`。
- 它会在 fusion-region 内做保守的 value 等价判断，包括：
  - 纯 op 结果等价
  - loop-carried iter_arg 与 `plt.scalar_out` 的直接递归等价
- 只在能证明等价时复用已有谓词，避免错误跨越 side effect 或复杂循环递归。

### 5.11 访存重定向与消除 (PTOFusionLoadStoreElision)

**设计动机 (Motivation)**
在已经形成稳定 A5VM carrier loop 的前提下，消除 fusion-region 内部仅用于中转的本地 store/load 往返，把 region-local 数据通路收缩到更接近寄存器/向量值直传的形态。

**流水线位置 (Pipeline Position)**

- **Pre-conditions**：完成低层循环融合、全局规范化和谓词消除。
- **Post-conditions**：局部 `a5vm.vsts -> a5vm.vlds` round-trip 被消除；非逃逸尾部 store 也可能被清理。

**核心逻辑与约束 (Logic & Constraints)**

- **输入契约**：处理对象是 `pto.fusion_region` 内的 A5VM post-lowering loop body，载体循环通常带 `llvm.loop.aivector_scope`。
- **核心行为**：
  - 归一化 tracked memref 根值，穿透 `bind_tile`、`memref.cast`、`reinterpret_cast`、`transpose` 等包装。
  - 以 `pto.yield` / region result 作为“外部可见性 frontier”，避免误删需要向 region 外暴露的写回。
  - 保守消除匹配的 store/load 对，并做 frontier-aware tail-store cleanup。
- **现实边界**：
  - 遇到无法做别名分析的内存 effect，会直接保守退出。
  - 仅处理 fusion-region 局部模式，不替代通用 DSE。

### 5.12 融合区域展平 (PTOFlattenFusionRegion)

**设计动机 (Motivation)**
一旦 region-local backend 优化全部结束，`pto.fusion_region` 这个结构化容器就不再需要，必须显式展平回父 block，恢复后端发射更容易消费的平面 A5VM IR。

**流水线位置 (Pipeline Position)**

- **Pre-conditions**：5.11 之后 region 内剩余 op 已经是最终 backend-ready 形式。
- **Post-conditions**：`pto.fusion_region` 和 `pto.yield` 被删除，父 block 只保留普通低层 op。

**核心逻辑与约束 (Logic & Constraints)**

- 把 region body 中除 terminator 外的 op 全部移到 wrapper 之前。
- 用 `pto.yield` 的 operands 替换 `pto.fusion_region` 的结果。
- 擦除 `pto.yield` 和 wrapper 本身。

### 5.13 A5VM backend 发射 (A5VM text / LLVM emission)

**设计动机 (Motivation)**
当前 tile fusion 主线的最终目标不是“生成 CCE IR 规划文档中的抽象出口”，而是进入现有 A5VM backend 发射器：要么打印/导出 A5VM 文本，要么继续走 LLVM emission，再交给后续工具链。

**流水线位置 (Pipeline Position)**

- **Pre-conditions**：IR 已经是展平后的 A5VM backend-ready 形式。
- **Post-conditions**：生成 A5VM 文本输出或 LLVM 级 backend 产物。

**核心逻辑与约束 (Logic & Constraints)**

- `llvm.loop.aivector_scope` 是 backend emitter 识别向量 section loop 的关键结构化标记。
- tile fusion 文档在这里需要关注的是“前面 pass 是否把 A5VM 结构准备对”，而不是再描述一个当前代码中不存在的 `PTOEmitCCEPass`。

## 6. 关键算法实现

- **预分析驱动的 block-local DFG 建模**：在 `tile_buf` 世界内提取 compute/local-boundary/hard-boundary、value liveness 和 write escape class，作为所有后续决策的统一依据。
- **保守 DAG 贪心规划**：当前默认策略是 `ConservativeDAGGreedyStrategyEngine + ConservativeDAGGreedyCostModel`，并非开放式插件系统。
- **span 压缩调度**：通过 barrier 分类与双向移动规则，把离散 group 压成连续 span，同时维持 SSA 和 boundary 合法性。
- **post-lowering stage matcher**：`PTOLowLevelLoopFusion` 在 `scf.for + a5vm.*` 层匹配同头循环和可重排 prelude，执行保守的相邻阶段融合。
- **frontier-aware cleanup**：`PTOFusionPredicateElision` 和 `PTOFusionLoadStoreElision` 都依赖 fusion-region frontier，避免把仍需对外可见的值错误消除。

## 7. 与系统其它模块的交互

- **shared pre-backend normalization**：tile fusion 前半段和 A5 backend 共用 `LoweringSyncToPipe`、`InferPTOLayout`、`PTOViewToMemref`、`PlanMemory`、`PTOInsertSync` 等 pass，文档必须把它们视作主线的一部分，而不是 region 外独立步骤。
- **A5VM lowering 契约**：`PTOA5VMVersionSelection` 决定 fusion-region 内外 lowering 变体，`PTOToA5VM` 决定后续低层循环优化所见的 IR 形态。
- **backend emitter 契约**：`A5VMTextEmitter` / `A5VMLLVMEmitter` 依赖 `llvm.loop.aivector_scope` 等结构化痕迹；前面 pass 若破坏这些结构，后端发射会直接受影响。
- **测试与 OpenSpec**：`test/tile_fusion/`、`test/samples/runop.sh -t TileFusion` 以及 `openspec/specs/*tile-fusion*` 是行为对齐的主要外部约束，不应再用历史 LibCall/CCE 设计作为 source of truth。

## 8. 性能验证与测试

### 8.1 功能验证

- pass 级回归优先使用 `test/tile_fusion/*.mlir`。
- 需要观察分组和调度结果时，可用：
  - `--test-only-op-scheduling`
  - `--test-only-fusion-region-gen`
- 需要验证 backend 主线形态时，优先检查 `--pto-backend=a5vm --enable-op-fusion --a5vm-print-ir` 输出。
- 动态 shape、local boundary、hard boundary、不可移动副作用 op 都应覆盖负例。

### 8.2 性能度量

- **结构指标**：
  - `pto.fusion.group_id/order` 是否符合预期。
  - `pto.fusion_region` 是否只包住一个连续 span。
  - `llvm.loop.aivector_scope` 是否仍在最终 carrier loop 上。
- **低层指标**：
  - `a5vm.plt_*` 冗余是否下降。
  - region 内 `vlds/vsts` round-trip 是否减少。
- **端到端样例**：
  - `python3 -m lit -sv test/tile_fusion`
  - `bash test/samples/runop.sh -t TileFusion`
  - 必要时用 `--print-ir-after-all` 对照关键阶段 IR。

## 9. 当前边界与后续方向

- **当前已落地边界**：
  - 主线只覆盖 A5/A5VM backend，不覆盖 EmitC。
  - 规划阶段仅支持保守的 block-local group，不做跨 basic block 融合。
  - 动态迭代域当前无法证明，相关链路会被拒绝。
  - `PreFusionAnalysis` 可识别的 compute family 比 `FusionPlan` 当前允许规划的 op 范围更宽。
- **后续可扩展方向**：
  - 把动态 `v_row/v_col` 证明补成独立 shape inference 主线。
  - 放宽 planner 对 reduce / broadcast 组合的实际落地支持。
  - 在保持 `pto.fusion_region` 契约稳定的前提下，继续增强 post-lowering loop fusion 和 load/store elimination 的覆盖面。
