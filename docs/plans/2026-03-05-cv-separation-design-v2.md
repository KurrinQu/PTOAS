# CV Separation Pass 设计方案（V2）

**日期**: 2026-03-05
**状态**: 已实现
**基于**: 原始设计 (2026-03-04) + 硬件数据通路修正 (信息补充.md)

---

## 1. 概述

### 1.1 目标

将 PTOAS 编译器中混合的 Cube/Vector 代码拆分为独立的 `pto.section.cube` 和 `pto.section.vector`，并在跨域数据依赖处插入 GM workspace 桥接和同步指令。

### 1.2 架构：两阶段 Pass

```
CVClassifyAndSplit → CVInsertBridge → [下游 pass pipeline]
```

- **CVClassifyAndSplit**：分类 Op 并拆分到 section
- **CVInsertBridge**：识别跨域依赖，插入 bridge 指令和同步

### 1.3 Pipeline 位置

通过 `--enable-cv-separation` 标志控制。开启后，两个 CV pass 插入到现有 pipeline 前端，**下游 pass 继续正常执行**（LoweringSyncToPipe → PlanMemory → EmitPTOManual → EmitC）。

```cpp
// ptoas.cpp pipeline
if (enableCVSeparation) {
  pm.addNestedPass<func::FuncOp>(pto::createCVClassifyAndSplitPass());
  pm.addNestedPass<func::FuncOp>(pto::createCVInsertBridgePass());
}
// 下游 pass 始终运行
pm.addNestedPass<func::FuncOp>(pto::createLoweringSyncToPipePass());
pm.addPass(pto::createPTOViewToMemrefPass());
// ... EmitC etc.
```

### 1.4 硬件数据通路

| 内存 | 地址空间 | 所属单元 | 数据通路 | Pipeline |
|------|---------|---------|---------|----------|
| GM | `gm` (1) | 共享 | 全局内存 | - |
| MAT (L1) | `mat` (2) | Cube | GM → MAT (MTE2) | PIPE_MTE2 |
| LEFT (L0A) | `left` (3) | Cube | MAT → LEFT (TMOV) | - |
| RIGHT (L0B) | `right` (4) | Cube | MAT → RIGHT (TMOV) | - |
| ACC (L0C) | `acc` (5) | Cube | matmul 输出 | PIPE_M |
| UB (UBUF) | `vec` (6) | Vector | GM → UB (MTE2) | PIPE_MTE2 |
| BIAS | `bias` (7) | Cube | 矩阵乘偏置 | - |
| SCALING | `scaling` (8) | Cube | 矩阵乘缩放 | - |

**Cube 侧完整数据通路**：
```
GM → MAT (via MTE2) → LEFT/RIGHT (via TMOV) → matmul → ACC → GM (via PIPE_FIX)
```

**Vector 侧完整数据通路**：
```
GM → UB (via MTE2) → VECTOR 计算 → UB → GM (via MTE3)
```

**跨域数据交换（当前版本）**：统一通过 GM workspace 中转。

---

## 2. CVClassifyAndSplit — Op 分类与拆分

### 2.1 Op 分类规则

分类采用**三级优先级**：

**第一级：Op 类型直接匹配**

| 域 | Op 类型 |
|----|--------|
| CUBE | `MatmulOp, MatmulAccOp, TMatmulOp, TMatmulAccOp, TMatmulBiasOp, TMatmulMxOp, TMatmulMxAccOp, TMatmulMxBiasOp, TGemvOp, TGemvAccOp, TGemvBiasOp` |
| VECTOR | `AddFOp, AddFDpsOp, TransOp, TTransOp, MovOp` |

> **注意**：`TMovOp` 不在上述列表中。TMOV 是灵活指令，其归属由第二级（地址空间）决定。

**第二级：地址空间匹配**

| 域 | 地址空间 |
|----|---------|
| CUBE | `LEFT, RIGHT, ACC, MAT, BIAS, SCALING` |
| VECTOR | `VEC` |

检查 op 所有 operand 和 result 的 memref 地址空间，任一匹配即归入对应域。

**TMovOp 分类示例**：
- `pto.tmov ins(memref<..., mat>) outs(memref<..., left>)` → 操作数含 MAT/LEFT → **CUBE**
- `pto.tmov ins(memref<..., vec>) outs(memref<..., vec>)` → 操作数含 VEC → **VECTOR**

**第三级：Fallback → SHARED**

不匹配以上任何规则的 op（如 `arith.constant`, `scf.yield`）归为 SHARED。

### 2.2 分类实现

```cpp
ComputeDomain classifyOp(Operation *op) {
  if (isa<SectionCubeOp, SectionVectorOp>(op))
    return ComputeDomain::SHARED;

  // 第一级：Op 类型
  if (isa<MatmulOp, MatmulAccOp, TMatmulOp, TMatmulAccOp,
          TMatmulBiasOp, TMatmulMxOp, TMatmulMxAccOp,
          TMatmulMxBiasOp, TGemvOp, TGemvAccOp, TGemvBiasOp>(op))
    return ComputeDomain::CUBE;

  if (isa<AddFOp, AddFDpsOp, TransOp, TTransOp, MovOp>(op))
    return ComputeDomain::VECTOR;

  // 第二级：地址空间
  static const pto::AddressSpace cubeSpaces[] = {
      pto::AddressSpace::LEFT, pto::AddressSpace::RIGHT,
      pto::AddressSpace::ACC, pto::AddressSpace::MAT,
      pto::AddressSpace::BIAS, pto::AddressSpace::SCALING};
  if (hasAddrSpaceIn(op, cubeSpaces))
    return ComputeDomain::CUBE;

  static const pto::AddressSpace vecSpaces[] = {pto::AddressSpace::VEC};
  if (hasAddrSpaceIn(op, vecSpaces))
    return ComputeDomain::VECTOR;

  return ComputeDomain::SHARED;
}
```

### 2.3 拆分算法

```
Step 1: 遍历 function body 中所有顶层 Op，标记 domain
Step 2: 对 SHARED Op（如 scf.for），递归分析 body：
        - body 仅含 CUBE → 整体归入 CUBE
        - body 仅含 VECTOR → 整体归入 VECTOR
        - body 混合 → 拆分为两个并行循环（同 trip count）
Step 3: 将连续同域 Op 合并并包裹为 pto.section.cube / pto.section.vector
Step 4: 保留已有的 section op 不变
```

**SectionOp 创建注意事项**：`builder.create<SectionCubeOp>(loc)` 创建的 region 有 0 个 block，必须手动 `region.emplaceBlock()` 后才能操作。

### 2.4 输入/输出 IR 示例

**输入（flat IR — Cube 完整数据通路）**：
```mlir
func.func @cube_full_path(
    %gm_a: memref<16x256xf16, #pto.address_space<gm>>,
    %gm_b: memref<256x16xf16, #pto.address_space<gm>>,
    %mat_a: memref<16x256xf16, #pto.address_space<mat>>,
    %mat_b: memref<256x16xf16, #pto.address_space<mat>>,
    %left: memref<16x256xf16, #pto.address_space<left>>,
    %right: memref<256x16xf16, #pto.address_space<right>>,
    %acc: memref<16x16xf32, #pto.address_space<acc>>) {
  pto.tload ins(%gm_a : ...) outs(%mat_a : ...)     // GM→MAT (MTE2)
  pto.tload ins(%gm_b : ...) outs(%mat_b : ...)     // GM→MAT (MTE2)
  pto.tmov ins(%mat_a : ...) outs(%left : ...)       // MAT→LEFT
  pto.tmov ins(%mat_b : ...) outs(%right : ...)      // MAT→RIGHT
  pto.tmatmul ins(%left, %right : ...) outs(%acc : ...)  // CUBE 计算
  return
}
```

**输出**：
```mlir
func.func @cube_full_path(...) {
  pto.section.cube {
    pto.tload ins(%gm_a : ...) outs(%mat_a : ...)   // GM→MAT
    pto.tload ins(%gm_b : ...) outs(%mat_b : ...)   // GM→MAT
    pto.tmov ins(%mat_a : ...) outs(%left : ...)     // MAT→LEFT
    pto.tmov ins(%mat_b : ...) outs(%right : ...)    // MAT→RIGHT
    pto.tmatmul ins(%left, %right : ...) outs(%acc : ...)
  }
  return
}
```

**TMOV 分类示例**：
```mlir
// 输入：两个 tmov，一个 cube 侧 (MAT→LEFT)，一个 vector 侧 (VEC→VEC)
func.func @tmov_classify(
    %mat_buf: memref<..., #pto.address_space<mat>>,
    %left: memref<..., #pto.address_space<left>>,
    %ub_src: memref<..., #pto.address_space<vec>>,
    %ub_dst: memref<..., #pto.address_space<vec>>) {
  pto.tmov ins(%mat_buf : ...) outs(%left : ...)     // MAT→LEFT → CUBE
  pto.tmov ins(%ub_src : ...) outs(%ub_dst : ...)    // VEC→VEC → VECTOR
  return
}

// 输出：
pto.section.cube {
  pto.tmov ins(%mat_buf : ...) outs(%left : ...)
}
pto.section.vector {
  pto.tmov ins(%ub_src : ...) outs(%ub_dst : ...)
}
```

**混合循环拆分示例**：
```mlir
// 输入：
scf.for %i = %c0 to %c4 step %c1 {
  pto.tmov ins(%mat_a : ...) outs(%left : ...)       // CUBE
  pto.tmatmul ins(%left, %right : ...) outs(%acc : ...)  // CUBE
  pto.tstore ins(%ub_buf : ...) outs(%gm_out : ...)  // VECTOR
}

// 输出：
pto.section.cube {
  scf.for %i = %c0 to %c4 step %c1 {
    pto.tmov ins(%mat_a : ...) outs(%left : ...)
    pto.tmatmul ins(%left, %right : ...) outs(%acc : ...)
  }
}
pto.section.vector {
  scf.for %i = %c0 to %c4 step %c1 {
    pto.tstore ins(%ub_buf : ...) outs(%gm_out : ...)
  }
}
```

---

## 3. CVInsertBridge — 数据桥接

### 3.1 职责

假设 workspace memref 已由 host 分配并作为 kernel 参数传入（约定：函数签名中最后一个 `address_space<gm>` 参数）。CVInsertBridge 负责：

1. 识别跨 section 的 SSA Value 依赖
2. 在生产者 section 末尾插入 `pto.tstore` 写入 workspace
3. 在消费者 section 开头插入 `pto.tload` 从 workspace 读取
4. 插入 `pto.sync.set` / `pto.sync.wait` 跨核同步
5. 替换消费者侧 SSA Value 引用

### 3.2 同步 Pipe 选择

根据生产者 section 类型选择不同的 pipeline：

| 生产者 Section | 数据通路 | sync.set pipe | sync.wait pipe |
|---------------|---------|---------------|----------------|
| SectionCubeOp | ACC → GM | **PIPE_FIX** | PIPE_MTE2 |
| SectionVectorOp | UB → GM | **PIPE_MTE3** | PIPE_MTE2 |

```cpp
auto storePipe = isa<SectionCubeOp>(bp.producerSection)
                     ? pto::PIPE::PIPE_FIX
                     : pto::PIPE::PIPE_MTE3;
```

### 3.3 Bridge 插入实现

```cpp
void insertBridge(BridgePoint &bp, Value workspace, unsigned flagId,
                  OpBuilder &builder) {
  // 1. Producer section 末尾：tstore + sync.set
  Block &prodBody = bp.producerSection->getRegion(0).front();
  builder.setInsertionPoint(&prodBody, prodBody.end());
  auto loc = bp.producerValue.getLoc();

  builder.create<TStoreOp>(loc, TypeRange{}, bp.producerValue, workspace);

  auto storePipe = isa<SectionCubeOp>(bp.producerSection)
                       ? pto::PIPE::PIPE_FIX
                       : pto::PIPE::PIPE_MTE3;
  auto pipeAttr = PipeAttr::get(builder.getContext(), storePipe);
  builder.create<SyncSetOp>(loc, pipeAttr, static_cast<uint32_t>(flagId));

  // 2. Consumer section 开头：sync.wait + tload
  Block &consBody = bp.consumerSection->getRegion(0).front();
  builder.setInsertionPointToStart(&consBody);

  auto waitPipe = PipeAttr::get(builder.getContext(), pto::PIPE::PIPE_MTE2);
  builder.create<SyncWaitOp>(loc, waitPipe, static_cast<uint32_t>(flagId));

  Value dst = bp.consumerUses[0]->get();
  builder.create<TLoadOp>(loc, TypeRange{}, workspace, dst);
}
```

### 3.4 DPS 格式说明

PTO dialect 中所有 Cube/Vector op（tload, tstore, tmov, tmatmul 等）都是 DPS (Destination-Passing Style) 格式：操作结果写入预分配的 memref buffer，**不产生逃逸的 SSA Value**。因此在当前 DPS IR 中，跨 section 的 SSA 依赖实际上不会发生，CVInsertBridge 主要用于处理未来可能出现的非 DPS 场景。

### 3.5 同步 Flag 管理

- Flag 编号按桥接点顺序递增：`flag[0]`, `flag[1]`, ...
- 同一个 SSA Value 被多个消费者使用时，只插入一次 tstore，每个消费者侧各一次 tload
- `sync.set` 表示数据已写入 GM，`sync.wait` 表示等待数据可读

---

## 4. 边界情况与错误处理

| 场景 | 处理方式 |
|------|---------|
| 无跨域依赖 | CVClassifyAndSplit 正常拆分，CVInsertBridge 无操作 |
| 纯 Cube / 纯 Vector 代码 | 只生成单个 section |
| 已有 section 的 IR | 保留已有 section，仅处理 loose ops |
| 混合域 scf.for | 克隆为两个循环，分别过滤保留对应域的 op |
| workspace 参数缺失 | 存在跨域依赖时 emit error 并中止 |
| TMOV 归属 | 由地址空间决定，非 op 类型 |

---

## 5. 测试用例

### 5.1 FileCheck 测试（验证 C++ 输出）

测试通过 `ptoas --enable-cv-separation` 运行全流程，CHECK 模式匹配最终 C++ 输出中的 `__DAV_CUBE__` / `__DAV_VEC__` 预处理守卫。

| 测试文件 | 验证内容 |
|---------|---------|
| `cv_pure_cube.mlir` | 完整 Cube 路径 (GM→MAT→LEFT/RIGHT→matmul) → 仅 `__DAV_CUBE__`，包含 TLOAD/TMOV/TMATMUL |
| `cv_classify_simple.mlir` | 纯 Vector ops → 仅 `__DAV_VEC__`，包含 TLOAD/TSTORE |
| `cv_tmov_classify.mlir` | TMOV 地址空间分类：MAT→LEFT → `__DAV_CUBE__`，VEC→VEC → `__DAV_VEC__` |
| `cv_split_loop.mlir` | 混合循环拆分 → 两个 for 循环分别在 CUBE/VEC 守卫内 |
| `cv_existing_sections.mlir` | 已有 section 保留 + loose op 归入新 section |
| `cv_no_cross_dep.mlir` | 无跨域依赖 → 无 set_flag/wait_flag |
| `cv_bridge_cube_to_vec.mlir` | Cube + Vector 分离（DPS 无 SSA 逃逸）|

### 5.2 示例：cv_tmov_classify.mlir

```mlir
// RUN: ptoas --enable-cv-separation %s | FileCheck %s

module {
  func.func @tmov_classify(
      %mat_buf: memref<16x256xf16, #pto.address_space<mat>>,
      %left: memref<16x256xf16, #pto.address_space<left>>,
      %ub_src: memref<16x256xf16, #pto.address_space<vec>>,
      %ub_dst: memref<16x256xf16, #pto.address_space<vec>>) {
    pto.tmov ins(%mat_buf : memref<16x256xf16, #pto.address_space<mat>>)
             outs(%left : memref<16x256xf16, #pto.address_space<left>>)
    pto.tmov ins(%ub_src : memref<16x256xf16, #pto.address_space<vec>>)
             outs(%ub_dst : memref<16x256xf16, #pto.address_space<vec>>)
    return
  }
}

// CHECK: __DAV_CUBE__
// CHECK: TMOV
// CHECK: __DAV_VEC__
// CHECK: TMOV
```

### 5.3 示例：cv_pure_cube.mlir

```mlir
// RUN: ptoas --enable-cv-separation %s | FileCheck %s

module {
  func.func @cube_full_path(
      %gm_a: memref<16x256xf16, #pto.address_space<gm>>,
      %gm_b: memref<256x16xf16, #pto.address_space<gm>>,
      %mat_a: memref<16x256xf16, #pto.address_space<mat>>,
      %mat_b: memref<256x16xf16, #pto.address_space<mat>>,
      %left: memref<16x256xf16, #pto.address_space<left>>,
      %right: memref<256x16xf16, #pto.address_space<right>>,
      %acc: memref<16x16xf32, #pto.address_space<acc>>) {
    pto.tload ins(%gm_a : ...) outs(%mat_a : ...)
    pto.tload ins(%gm_b : ...) outs(%mat_b : ...)
    pto.tmov ins(%mat_a : ...) outs(%left : ...)
    pto.tmov ins(%mat_b : ...) outs(%right : ...)
    pto.tmatmul ins(%left, %right : ...) outs(%acc : ...)
    return
  }
}

// CHECK: __DAV_CUBE__
// CHECK: TLOAD
// CHECK: TLOAD
// CHECK: TMOV
// CHECK: TMOV
// CHECK: TMATMUL
// CHECK-NOT: __DAV_VEC__
```

---

## 6. 涉及文件

| 文件 | 用途 |
|------|------|
| `lib/PTO/Transforms/CVClassifyAndSplit.cpp` | Pass 1: Op 分类与 section 拆分 |
| `lib/PTO/Transforms/CVInsertBridge.cpp` | Pass 2: 跨域依赖桥接 |
| `include/PTO/Transforms/Passes.td` | Pass 定义（TableGen）|
| `include/PTO/Transforms/Passes.h` | Pass 工厂函数声明 |
| `lib/PTO/Transforms/CMakeLists.txt` | 构建配置 |
| `tools/ptoas/ptoas.cpp` | Pipeline 集成 (`--enable-cv-separation`) |
| `test/basic/cv_*.mlir` | 7 个 FileCheck 测试 |

---

## 7. 后续优化（不在当前版本范围）

- A5 架构片上通路（ACC→VEC, UB→MAT）替代 GM workspace
- Workspace double-buffering
- 跨域 acc 类型自动 cast（float32 → float16）减少传输量
- 嵌套循环递归拆分
