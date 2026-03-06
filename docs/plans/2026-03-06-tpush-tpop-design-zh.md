# TPUSH/TPOP PTOAS 编译器支持设计

## 1. 概述

本文档定义 PTOAS 编译器侧对 `TPUSH`/`TPOP` 指令的支持设计。TPUSH/TPOP 是一种基于 ring buffer 的 cluster 内 Cube 与 Vector 核间数据通信机制。

TPUSH/TPOP 替代 `CVInsertBridge` 中现有的 GM bridge（TStore→GM→TLoad），作为跨 section 的主要数据传输通道。通过平台无关的 IR ops 同时支持 A2/A3（ring buffer 位于 GM）和 A5（ring buffer 位于消费者片上 SRAM，zero-copy）平台，平台差异在 EmitC lowering 阶段根据 `--pto-arch` 处理。

### 参考文档

- ISA 层设计：`HL_ptoisa_newfeature20260306_TPUSH_TPOP.md`
- CV 分离设计：`docs/plans/2026-03-05-cv-separation-design-v2.md`

---

## 2. 新增 Op 定义（7 个 Op）

所有 op 遵循现有 PTOAS 约定：side-effect 模型（无 SSA 状态线程化）、通过 `OpPipeInterface` 支持同步插入、通过 `MemoryEffectsOpInterface` 保证优化安全性。

### 2.1 内存声明 Op

#### `pto.reserve_buffer`

在消费者 InCore 函数的本地 SRAM 中声明一段保留区域，用于 ring buffer slot。定义在**函数级别**（section 外部），使 `section_cube` 和 `section_vector` 均可引用其结果。

```tablegen
def ReserveBufferOp : PTO_Op<"reserve_buffer", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Declare a reserved SRAM region for ring buffer slots";
    let arguments = (ins
        StrAttr:$name,                          // 缓冲区标识名
        I32Attr:$size,                          // 总字节数 = SLOT_NUM * SLOT_SIZE
        OptionalAttr<I32Attr>:$base,            // 基地址（缺省 = 编译器自动分配）
        PTO_AddressSpaceAttr:$memory_space      // VEC (Vector UB) 或 MAT (Cube L1)
    );
    let results = (outs I32:$result);           // 解析后的基地址（编译期常量）
    let hasVerifier = 1;
}
```

**Verifier 规则：**
- 必须位于函数级别（不在 `SectionCubeOp` 或 `SectionVectorOp` 内部）
- `size` > 0
- `memory_space` 必须为 `VEC` 或 `MAT`
- 若指定了 `base`，必须满足硬件对齐要求（如 32 字节对齐）

**语义：**
- A5 平台：在消费者 SRAM 中保留一段区域，`PTOPlanMemory` 解析地址
- A2A3 平台：op 仍会生成以保持一致性，但解析后的地址运行时不使用（ring buffer 通过 `gm_slot_buffer` 位于 GM）。lowering 阶段可将其折叠为常量 0

### 2.2 初始化 Ops

#### `pto.aic_initialize_pipe`

Cube 核启动时调用一次，初始化指定方向的 ring buffer 管道。定义在**函数级别**。

```tablegen
def AicInitializePipeOp : PTO_Op<"aic_initialize_pipe", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Initialize ring buffer pipe(s) on Cube core";
    let arguments = (ins
        I8Attr:$dir_mask,                       // 0b01=C2V, 0b10=V2C, 0b11=双向
        I32Attr:$slot_size,                     // 每个 slot 字节数（= Tile 大小）
        PTODpsType:$gm_slot_buffer,             // GM 缓冲区（A2A3 有效；A5 为 nullptr）
        I32:$c2v_consumer_buf,                  // C2V 方向消费者 SRAM 基地址
        I32:$v2c_consumer_buf                   // V2C 方向消费者 SRAM 基地址
    );
    let results = (outs);
    let hasVerifier = 1;
}
```

#### `pto.aiv_initialize_pipe`

Vector 核启动时调用一次，签名相同。

```tablegen
def AivInitializePipeOp : PTO_Op<"aiv_initialize_pipe", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Initialize ring buffer pipe(s) on Vector core";
    let arguments = (ins
        I8Attr:$dir_mask,
        I32Attr:$slot_size,
        PTODpsType:$gm_slot_buffer,
        I32:$c2v_consumer_buf,
        I32:$v2c_consumer_buf
    );
    let results = (outs);
    let hasVerifier = 1;
}
```

**共同 Verifier 规则：**
- 必须位于函数级别（不在任何 section 内部）
- `dir_mask` 必须为 `1`（C2V）、`2`（V2C）或 `3`（双向）
- `slot_size` > 0 且满足硬件对齐要求
- `c2v_consumer_buf` 和 `v2c_consumer_buf` 通常来自 `reserve_buffer` 的结果

### 2.3 数据传输 Ops

#### `pto.tpush_to_aiv`（Cube 生产者，C2V 方向）

```tablegen
def TPushToAivOp : PTO_TOp<"tpush_to_aiv", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Push tile from Cube to buddy Vector via ring buffer";
    let arguments = (ins
        PTODpsType:$tile,                       // 源 tile 数据
        I8Attr:$aiv_idx                         // 目标 buddy Vector 核索引（0 或 1）
    );
    let results = (outs);
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE1; }
    }];
}
```

#### `pto.tpush_to_aic`（Vector 生产者，V2C 方向）

```tablegen
def TPushToAicOp : PTO_TOp<"tpush_to_aic", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Push tile from Vector to buddy Cube via ring buffer";
    let arguments = (ins
        PTODpsType:$tile,                       // 源 tile 数据
        I8Attr:$aiv_idx                         // 本 Vector 核自身索引（0 或 1）
    );
    let results = (outs);
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE2; }
    }];
}
```

#### `pto.tpop_from_aic`（Vector 消费者，C2V 方向）

```tablegen
def TPopFromAicOp : PTO_TOp<"tpop_from_aic", [
    PTO_DpsInitOpInterface,
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Pop tile that Cube pushed, into destination tile buffer";
    let arguments = (ins
        PTODpsType:$tile,                       // 目标 tile buffer（DPS init）
        I8Attr:$aiv_idx                         // 本 Vector 核自身索引（0 或 1）
    );
    let results = (outs);
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE1; }
        ::mlir::MutableOperandRange getDpsInitsMutable() { return getTileMutable(); }
    }];
}
```

#### `pto.tpop_from_aiv`（Cube 消费者，V2C 方向）

```tablegen
def TPopFromAivOp : PTO_TOp<"tpop_from_aiv", [
    PTO_DpsInitOpInterface,
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Pop tile that Vector pushed, into destination tile buffer";
    let arguments = (ins
        PTODpsType:$tile,                       // 目标 tile buffer（DPS init）
        I8Attr:$aiv_idx                         // 源 buddy Vector 核索引（0 或 1）
    );
    let results = (outs);
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE2; }
        ::mlir::MutableOperandRange getDpsInitsMutable() { return getTileMutable(); }
    }];
}
```

**数据传输 ops 共同 Verifier 规则：**
- `aiv_idx` 必须为 0 或 1
- `tpush_to_aiv` / `tpop_from_aiv` 必须位于 `SectionCubeOp` 内
- `tpush_to_aic` / `tpop_from_aic` 必须位于 `SectionVectorOp` 内
- `tile` 类型必须为 `TileBufType`

### 2.4 Op 总览

| Op | 位置 | 核类型 | 角色 | Pipe | DPS |
|---|---|---|---|---|---|
| `pto.reserve_buffer` | 函数级别 | — | 内存声明 | 无 | 否（返回 i32） |
| `pto.aic_initialize_pipe` | 函数级别 | Cube | 初始化 | 无 | 否 |
| `pto.aiv_initialize_pipe` | 函数级别 | Vector | 初始化 | 无 | 否 |
| `pto.tpush_to_aiv` | SectionCubeOp | Cube | 生产者（C2V） | PIPE_MTE1 | 否（读取 tile） |
| `pto.tpush_to_aic` | SectionVectorOp | Vector | 生产者（V2C） | PIPE_MTE2 | 否（读取 tile） |
| `pto.tpop_from_aic` | SectionVectorOp | Vector | 消费者（C2V） | PIPE_MTE1 | 是（写入 tile） |
| `pto.tpop_from_aiv` | SectionCubeOp | Cube | 消费者（V2C） | PIPE_MTE2 | 是（写入 tile） |

> **注意：** PIPE 分配（MTE1/MTE2）为初始建议，实际管道映射需根据达芬奇架构硬件 MTE 通道约束确认。

---

## 3. 与 CV Separation 的集成

### 3.1 当前 CV Separation 流程（已合并）

```
Mixed InCore 函数（Cube + Vector ops 交错）
        |
        v
  CVClassifyAndSplit
  -- 按硬件域分类 ops
  -- 包裹进 SectionCubeOp / SectionVectorOp 区域
        |
        v
  CVInsertBridge
  -- 检测跨域数据依赖
  -- 插入 GM workspace bridge：TStore(src -> GM) + TLoad(GM -> dst)
        |
        v
  PTOInsertSync -> PTOPlanMemory -> PTOToEmitC
```

### 3.2 集成 TPUSH/TPOP 后的新流程

```
Mixed InCore 函数
        |
        v
  CVClassifyAndSplit                 <-- 不变
        |
        v
  CVInsertBridge                     <-- 扩展：生成 tpush/tpop 替代 GM bridge
  -- 检测跨域数据依赖
  -- 对每条依赖边：生成 tpush/tpop 对
  -- 在函数级别生成 reserve_buffer（A5 消费者 SRAM 保留）
  -- 在函数级别生成 initialize_pipe
        |
        v
  PTOInsertSync                      <-- 不变（通过 OpPipeInterface 自动适配）
        |
        v
  PTOPlanMemory                      <-- 扩展：处理 reserve_buffer
        |
        v
  PTOToEmitC                         <-- 扩展：7 个新 lowering pattern
```

### 3.3 CVInsertBridge 扩展

#### 当前行为（GM bridge）

对每条跨域数据依赖 `%v`（在 SectionA 中定义，在 SectionB 中使用）：
1. 在生产者 section 中插入 `pto.tstore %v -> %gm_workspace`
2. 在消费者 section 中插入 `pto.tload %gm_workspace -> %v_copy`
3. 将消费者 section 中对 `%v` 的使用替换为 `%v_copy`

#### 新行为（TPUSH/TPOP ring buffer）

对每条跨域数据依赖 `%v`：

**步骤 1 — 在函数级别生成设置代码（每个函数一次，位于所有 section 之前）：**

```mlir
// Ring buffer 声明（用于 A5 消费者 SRAM 保留）
%buf_c2v = pto.reserve_buffer {
    name = "c2v_slot_buffer",
    size = SLOT_NUM * SLOT_SIZE,
    memory_space = VEC                  // 或 MAT，取决于消费者核类型
}

// 两侧初始化
pto.aic_initialize_pipe {dir_mask = 1, slot_size = SLOT_SIZE}
    (%gm_slot_buf, %buf_c2v, %cst0)
pto.aiv_initialize_pipe {dir_mask = 1, slot_size = SLOT_SIZE}
    (%gm_slot_buf, %buf_c2v, %cst0)
```

双向通信场景（同时存在 C2V 和 V2C 依赖）：

```mlir
%buf_c2v = pto.reserve_buffer {name = "c2v", size = ..., memory_space = VEC}
%buf_v2c = pto.reserve_buffer {name = "v2c", size = ..., memory_space = MAT}

pto.aic_initialize_pipe {dir_mask = 3, slot_size = SLOT_SIZE}
    (%gm_slot_buf, %buf_c2v, %buf_v2c)
pto.aiv_initialize_pipe {dir_mask = 3, slot_size = SLOT_SIZE}
    (%gm_slot_buf, %buf_c2v, %buf_v2c)
```

**步骤 2 — 在生产者 section 中生成 tpush：**

```mlir
pto.section_cube {
    ...
    pto.tpush_to_aiv(%v) {aiv_idx = 0}    // 推送跨域数据
}
```

**步骤 3 — 在消费者 section 中生成 tpop：**

```mlir
pto.section_vector {
    %v_recv = pto.alloc_tile ...
    pto.tpop_from_aic(%v_recv) {aiv_idx = 0}
    // 将 %v 的所有使用替换为 %v_recv
    ...
}
```

#### SLOT_SIZE 和 SLOT_NUM 的确定

`CVInsertBridge` 从跨域 tile 计算这些参数：

- `SLOT_SIZE` = 传输 tile 的字节大小（`TileBufType` shape * 元素大小）
- `SLOT_NUM` = 8（单向）或 4（双向），由 `dir_mask` 决定

#### 多条跨域依赖

同一方向有多个 tile 值跨越域边界时，它们通过同一 ring buffer 通道串行传输。生产者按顺序推送，消费者按相同顺序弹出（FIFO 保证）。

若 tile 在**两个方向**同时跨越（Cube->Vector 和 Vector->Cube），使用双向模式（`dir_mask = 3`，`SLOT_NUM = 4`）。

### 3.4 PTOPlanMemory 扩展

`PTOPlanMemory` 需处理 `reserve_buffer` ops：

```
对函数中每个 reserve_buffer op：

    若 base 属性存在（手动指定）：
        验证 [base, base + size) 与已有分配无重叠
        标记区域为已占用
        将 op 结果解析为字面量 base 值

    若 base 属性缺省（auto 模式）：
        加入待分配列表

分配所有普通 tile 和临时变量地址，跳过已占用区域。

对每个待分配的 auto reserve_buffer：
    在对应 SRAM 中找到所需大小的空闲区域
    分配地址，标记为已占用
    将 op 结果解析为分配的地址（常量折叠）
```

**地址分配器约定：** `[BASE, BASE + SIZE)` 区间为保留区域，其他 tile、临时变量或溢出分配均不得与之重叠。

### 3.5 PTOInsertSync — 无需修改

`PTOInsertSync` pass 已通过 `OpPipeInterface` 工作：

- `tpush_to_aiv` 声明 `PIPE_MTE1` → sync 插入自动处理 MTE1 边界
- `tpop_from_aic` 声明 `PIPE_MTE1` → 同上
- `tpush_to_aic` 声明 `PIPE_MTE2` → sync 插入自动处理 MTE2 边界
- `tpop_from_aiv` 声明 `PIPE_MTE2` → 同上

跨核同步（Cube 与 Vector 之间的 flag SET/WAIT）是 tpush/tpop **内部语义**的一部分，在 EmitC lowering 时生成。`PTOInsertSync` 仅负责同一核内不同 pipe 之间的同步。

### 3.6 PTOToEmitC Lowering

7 个新的 conversion pattern：

| Pattern | 输入 Op | EmitC 输出 |
|---|---|---|
| `ReserveBufferToEmitC` | `pto.reserve_buffer` | 消除（由 PlanMemory 解析为 `emitc::ConstantOp`） |
| `AicInitPipeToEmitC` | `pto.aic_initialize_pipe` | `emitc::CallOpaqueOp("aic_initialize_pipe", dir_mask, slot_size, gm_buf, c2v_buf, v2c_buf)` |
| `AivInitPipeToEmitC` | `pto.aiv_initialize_pipe` | `emitc::CallOpaqueOp("aiv_initialize_pipe", dir_mask, slot_size, gm_buf, c2v_buf, v2c_buf)` |
| `TPushToAivToEmitC` | `pto.tpush_to_aiv` | `emitc::CallOpaqueOp("tpush_to_aiv", tile, aiv_idx)` |
| `TPushToAicToEmitC` | `pto.tpush_to_aic` | `emitc::CallOpaqueOp("tpush_to_aic", tile, aiv_idx)` |
| `TPopFromAicToEmitC` | `pto.tpop_from_aic` | `emitc::CallOpaqueOp("tpop_from_aic", tile, aiv_idx)` |
| `TPopFromAivToEmitC` | `pto.tpop_from_aiv` | `emitc::CallOpaqueOp("tpop_from_aiv", tile, aiv_idx)` |

EmitC 输出直接映射到 pto-isa C++ 库函数。平台特定行为（GM DMA 与 SRAM zero-copy）由 pto-isa 库在运行时根据 `PLATFORM_ID` 处理。

---

## 4. 端到端示例

### 4.1 输入：Mixed InCore kernel（用户编写）

```python
@pl.incore
def matmul_relu_kernel(gm_a, gm_b, gm_c):
    tile_a = pl.tload(gm_a)
    tile_c = pl.tmatmul(tile_a, tile_b)    # Cube 操作
    tile_out = pl.trelu(tile_c)            # Vector 操作 -- 跨域依赖
    pl.tstore(tile_out, gm_c)
```

### 4.2 CVClassifyAndSplit 后

```mlir
func @matmul_relu_kernel(%gm_a, %gm_b, %gm_c, %gm_slot_buf) {
    pto.section_cube {
        %a = pto.alloc_tile ...
        pto.tload(%gm_a, %a)
        %c = pto.alloc_tile ...
        pto.tmatmul(%a, %b, %c)
        // %c 被 vector section 使用 -- 跨域依赖
    }
    pto.section_vector {
        %out = pto.alloc_tile ...
        pto.trelu(%c, %out)              // %c 来自 cube section
        pto.tstore(%out, %gm_c)
    }
}
```

### 4.3 CVInsertBridge 扩展后（TPUSH/TPOP）

```mlir
func @matmul_relu_kernel(%gm_a, %gm_b, %gm_c, %gm_slot_buf) {
    // --- 函数级别：ring buffer 设置 ---
    %cst0 = arith.constant 0 : i32
    %buf = pto.reserve_buffer {
        name = "c2v_slot_buffer",
        size = 32768,                       // 8 slots * 4096 字节
        memory_space = VEC
    }
    pto.aic_initialize_pipe {dir_mask = 1 : i8, slot_size = 4096 : i32}
        (%gm_slot_buf, %buf, %cst0)
    pto.aiv_initialize_pipe {dir_mask = 1 : i8, slot_size = 4096 : i32}
        (%gm_slot_buf, %buf, %cst0)

    // --- Cube section ---
    pto.section_cube {
        %a = pto.alloc_tile ...
        pto.tload(%gm_a, %a)
        %c = pto.alloc_tile ...
        pto.tmatmul(%a, %b, %c)
        pto.tpush_to_aiv(%c) {aiv_idx = 0 : i8}
    }

    // --- Vector section ---
    pto.section_vector {
        %recv = pto.alloc_tile ...
        pto.tpop_from_aic(%recv) {aiv_idx = 0 : i8}
        %out = pto.alloc_tile ...
        pto.trelu(%recv, %out)
        pto.tstore(%out, %gm_c)
    }
}
```

### 4.4 PTOPlanMemory 后

```mlir
// reserve_buffer 解析：%buf 折叠为常量 0x1000
// alloc_tile 地址分配避开 [0x1000, 0x9000)
// 所有 tile 地址均为已解析常量
```

### 4.5 PTOToEmitC 后（最终 C++ 输出）

```cpp
// === Cube kernel ===
void cube_kernel(__gm__ half* gm_a, __gm__ half* gm_b,
                 __gm__ half* gm_c, __gm__ void* gm_slot_buf) {
    aic_initialize_pipe(DIR_C2V, 4096, gm_slot_buf, 0x1000, 0);

    TLOAD(tile_a, gm_a_partition);
    TMATMUL(tile_c, tile_a, tile_b);
    tpush_to_aiv(tile_c, 0);
}

// === Vector kernel ===
void vector_kernel(__gm__ half* gm_a, __gm__ half* gm_b,
                   __gm__ half* gm_c, __gm__ void* gm_slot_buf) {
    aiv_initialize_pipe(DIR_C2V, 4096, gm_slot_buf, 0x1000, 0);

    tpop_from_aic(tile_recv, 0);
    TRELU(tile_out, tile_recv);
    TSTORE(gm_c_partition, tile_out);
}
```

### 4.6 双向示例（Cube->Vector + Vector->Cube）

```mlir
func @bidir_kernel(%gm_a, %gm_c, %gm_slot_buf) {
    %cst0 = arith.constant 0 : i32
    %buf_c2v = pto.reserve_buffer {name="c2v", size=16384, memory_space=VEC}
    %buf_v2c = pto.reserve_buffer {name="v2c", size=16384, memory_space=MAT}
    pto.aic_initialize_pipe {dir_mask=3:i8, slot_size=4096:i32}
        (%gm_slot_buf, %buf_c2v, %buf_v2c)
    pto.aiv_initialize_pipe {dir_mask=3:i8, slot_size=4096:i32}
        (%gm_slot_buf, %buf_c2v, %buf_v2c)

    pto.section_cube {
        pto.tpush_to_aiv(%matmul_result) {aiv_idx = 0 : i8}    // C2V
        %preprocessed = pto.alloc_tile ...
        pto.tpop_from_aiv(%preprocessed) {aiv_idx = 0 : i8}    // V2C
    }
    pto.section_vector {
        %recv = pto.alloc_tile ...
        pto.tpop_from_aic(%recv) {aiv_idx = 0 : i8}            // C2V
        pto.tpush_to_aic(%loaded_data) {aiv_idx = 0 : i8}      // V2C
    }
}
```

---

## 5. 需修改的文件

| 层次 | 文件 | 修改内容 |
|---|---|---|
| **ODS** | `include/PTO/IR/PTOOps.td` | 新增 7 个 op 定义 |
| **C++ IR** | `lib/PTO/IR/PTO.cpp` | 新增 7 个 op 的 verifier 实现 |
| **CV Bridge** | `lib/PTO/Transforms/CVInsertBridge.cpp` | 扩展：生成 tpush/tpop + reserve_buffer + initialize_pipe |
| **PlanMemory** | `lib/PTO/Transforms/PTOPlanMemory.cpp` | 处理 reserve_buffer：保留、自动分配、常量折叠 |
| **EmitC** | `lib/PTO/Transforms/PTOToEmitC.cpp` | 新增 7 个 conversion pattern |
| **Python** | `python/pto/dialects/PTOOps.td` | 自动生成（wrapper 包含 PTOOps.td） |
| **测试** | `test/basic/tpush_tpop_*.mlir` | ops、verifier 和 lowering 的回归测试 |

---

## 6. 设计决策总结

| 决策点 | 选择 | 理由 |
|---|---|---|
| 状态模型 | Side-effect（无 SSA 状态线程化） | 与现有 set_flag/wait_flag/get_buf/rls_buf 保持一致 |
| 平台处理 | 单一 op 集合，arch 感知的 lowering | 匹配现有模式（TLoadOp 等），避免 op 膨胀 |
| 缓冲区类型 | 不引入新类型；reserve_buffer 返回 i32 | 最小化 IR 复杂度；ring buffer 状态仅存在于 lowered C++ 中 |
| Op 位置 | reserve_buffer + initialize_pipe 在函数级别；tpush/tpop 在 section 内 | MLIR SSA 作用域：函数级别值对嵌套 region 可见 |
| import_peer_buffer | 移除 | 不需要 — 两个 section 在同一函数内，可共享 SSA 值 |
| 方向编码 | 编码在操作码中（4 个不同 op） | 匹配 ISA 设计；支持按核类型的静态 verifier 检查 |
| CV bridge 集成 | 扩展 CVInsertBridge | 复用现有跨域依赖检测；tpush/tpop 替代 GM bridge |
| InsertSync | 无需修改 | OpPipeInterface 自动适配；跨核同步是 tpush/tpop 内部语义 |
