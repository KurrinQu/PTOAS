# CV Separation Pass 实施方案（V2）

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现两个 MLIR pass（CVClassifyAndSplit + CVInsertBridge），将混合 Cube/Vector IR 拆分为并行 section，并在跨域依赖处插入 GM workspace 桥接。

**Architecture:** 两阶段 pass — Phase 1 按域分类 op 并包裹 section；Phase 2 识别跨域 SSA 依赖并插入 TStore/TLoad 桥接和 sync 同步。通过 `--enable-cv-separation` flag 控制，下游 pass pipeline 正常执行。

**Tech Stack:** MLIR C++ (PassWrapper, OpBuilder), PTO dialect ops (SectionCubeOp, SectionVectorOp, TLoadOp, TStoreOp, SyncSetOp, SyncWaitOp), SCF dialect (scf.for)

**Design doc:** `docs/plans/2026-03-05-cv-separation-design-v2.md`

---

### Task 1: CVClassifyAndSplit — Pass 骨架与注册

**Files:**
- Create: `lib/PTO/Transforms/CVClassifyAndSplit.cpp`
- Modify: `include/PTO/Transforms/Passes.td`
- Modify: `include/PTO/Transforms/Passes.h`
- Modify: `lib/PTO/Transforms/CMakeLists.txt`

**Step 1: Add pass definition to Passes.td**

```tablegen
def CVClassifyAndSplit : Pass<"pto-cv-classify-and-split", "func::FuncOp"> {
  let summary = "Classify ops into Cube/Vector domains and split into sections";
  let constructor = "mlir::pto::createCVClassifyAndSplitPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect", "mlir::func::FuncDialect", "mlir::scf::SCFDialect"
  ];
}
```

**Step 2: Add factory declaration to Passes.h**

```cpp
std::unique_ptr<Pass> createCVClassifyAndSplitPass();
```

**Step 3: Add source to CMakeLists.txt**

```cmake
CVClassifyAndSplit.cpp
```

**Step 4: Create pass skeleton**

```cpp
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::pto;

namespace {
enum class ComputeDomain { CUBE, VECTOR, SHARED };

class CVClassifyAndSplitPass
    : public PassWrapper<CVClassifyAndSplitPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CVClassifyAndSplitPass)
  StringRef getArgument() const override { return "pto-cv-classify-and-split"; }
  StringRef getDescription() const override {
    return "Classify ops into Cube/Vector domains and split into sections";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pto::PTODialect, func::FuncDialect, scf::SCFDialect>();
  }
  void runOnOperation() override { /* TODO */ }
};
} // namespace

namespace mlir { namespace pto {
std::unique_ptr<Pass> createCVClassifyAndSplitPass() {
  return std::make_unique<CVClassifyAndSplitPass>();
}
}} // namespace
```

**Step 5: Build**
```bash
cd build && cmake --build . --target ptoas -j$(sysctl -n hw.ncpu)
```

**Step 6: Commit**
```bash
git commit -m "feat(cv-split): add CVClassifyAndSplit pass skeleton"
```

---

### Task 2: CVClassifyAndSplit — Op 分类逻辑

**Files:**
- Modify: `lib/PTO/Transforms/CVClassifyAndSplit.cpp`
- Create: `test/basic/cv_classify_simple.mlir`

**Step 1: Implement classifyOp()**

三级优先级分类：
1. Op 类型匹配 — Cube: matmul 系列; Vector: AddF/Trans/Mov（**不含 TMovOp**）
2. 地址空间匹配 — Cube: `LEFT, RIGHT, ACC, MAT, BIAS, SCALING`; Vector: `VEC`
3. Fallback → SHARED

辅助函数 `getAddrSpace()` 和 `hasAddrSpaceIn()` 检查 operand/result 的 memref 地址空间。

**Step 2: Create test**

```mlir
// RUN: ptoas --enable-cv-separation %s | FileCheck %s
// 纯 vector ops → 仅 __DAV_VEC__
// CHECK: __DAV_VEC__
// CHECK: TLOAD
// CHECK: TSTORE
// CHECK-NOT: __DAV_CUBE__
```

**Step 3: Commit**
```bash
git commit -m "feat(cv-split): implement op classification logic"
```

---

### Task 3: CVClassifyAndSplit — Section 包裹逻辑

**Files:**
- Modify: `lib/PTO/Transforms/CVClassifyAndSplit.cpp`

**Step 1: Implement runOnOperation()**

1. 遍历 function body，分类每个 top-level op
2. 对 scf.for 递归判断 `classifyRegion()`
3. 连续同域 op 合并包裹为 section
4. **关键**：SectionOp 创建后 region 为空，必须 `region.emplaceBlock()`
5. 已有 section op 保留不动（归为 SHARED）

**Step 2: Build and test**

**Step 3: Commit**
```bash
git commit -m "feat(cv-split): implement section wrapping"
```

---

### Task 4: CVClassifyAndSplit — 混合循环拆分

**Files:**
- Modify: `lib/PTO/Transforms/CVClassifyAndSplit.cpp`
- Create: `test/basic/cv_split_loop.mlir`

**Step 1: Implement splitMixedLoop()**

当 `classifyRegion()` 返回 SHARED（混合域）且 op 是 `scf.ForOp` 时：
1. 克隆循环两次 (cubeLoop, vecLoop)
2. `filterLoopBody()` 移除不属于目标域的 op
3. 各包裹进 SectionCubeOp / SectionVectorOp
4. 删除原循环

**Step 2: Create test**

```mlir
// CHECK: __DAV_CUBE__
// CHECK: for
// CHECK: TMOV
// CHECK: TMATMUL
// CHECK: __DAV_VEC__
// CHECK: for
// CHECK: TSTORE
```

**Step 3: Commit**
```bash
git commit -m "feat(cv-split): handle mixed-domain loop splitting"
```

---

### Task 5: CVInsertBridge — Pass 骨架与注册

**Files:**
- Create: `lib/PTO/Transforms/CVInsertBridge.cpp`
- Modify: `include/PTO/Transforms/Passes.td`
- Modify: `include/PTO/Transforms/Passes.h`
- Modify: `lib/PTO/Transforms/CMakeLists.txt`

同 Task 1 模式创建骨架。Pass 名 `pto-cv-insert-bridge`，工厂函数 `createCVInsertBridgePass()`。

**Commit**
```bash
git commit -m "feat(cv-bridge): add CVInsertBridge pass skeleton"
```

---

### Task 6: CVInsertBridge — 跨域依赖识别

**Files:**
- Modify: `lib/PTO/Transforms/CVInsertBridge.cpp`

**Step 1: Implement findBridgePoints()**

`func.walk()` 遍历所有 consumer op，对每个 operand 检查其 `getDefiningOp()` 是否在不同 section 中。收集为 `BridgePoint` 四元组。同一 Value 多个消费者去重。

**Step 2: Implement getEnclosingSection()**

沿 `getParentOp()` 链向上查找 SectionCubeOp/SectionVectorOp。

**Commit**
```bash
git commit -m "feat(cv-bridge): implement cross-domain dependency detection"
```

---

### Task 7: CVInsertBridge — Bridge 插入

**Files:**
- Modify: `lib/PTO/Transforms/CVInsertBridge.cpp`
- Create: `test/basic/cv_bridge_cube_to_vec.mlir`

**Step 1: Implement findWorkspaceArg()**

约定：函数签名中最后一个 `address_space<gm>` 参数作为 workspace。

**Step 2: Implement insertBridge()**

- Producer section 末尾：`TStoreOp(loc, TypeRange{}, producerValue, workspace)` + `SyncSetOp(loc, pipeAttr, flagId)`
- Consumer section 开头：`SyncWaitOp(loc, waitPipe, flagId)` + `TLoadOp(loc, TypeRange{}, workspace, dst)`
- **Pipe 选择**：
  - Producer 是 SectionCubeOp → `PIPE_FIX`（ACC→GM）
  - Producer 是 SectionVectorOp → `PIPE_MTE3`（UB→GM）
  - sync.wait 统一 `PIPE_MTE2`

**Step 3: Update runOnOperation()**

调用 `findBridgePoints()` → 若非空则 `findWorkspaceArg()` → 逐个 `insertBridge()`。

**Commit**
```bash
git commit -m "feat(cv-bridge): implement bridge insertion with correct pipe selection"
```

---

### Task 8: Pipeline 集成

**Files:**
- Modify: `tools/ptoas/ptoas.cpp`

**Step 1: Add CLI flag**

```cpp
static llvm::cl::opt<bool> enableCVSeparation(
    "enable-cv-separation",
    llvm::cl::desc("Enable Cube/Vector separation passes"),
    llvm::cl::init(false));
```

**Step 2: Register passes**

在 main() 中 `ParseCommandLineOptions` 之前：
```cpp
::registerCVClassifyAndSplit();
::registerCVInsertBridge();
```

注意：不要用 `pto::registerPTOPasses()`（部分 pass 未实现 `getArgument()` 会导致崩溃），改为单独注册。

**Step 3: Insert CV passes into pipeline**

```cpp
if (enableCVSeparation) {
  pm.addNestedPass<func::FuncOp>(pto::createCVClassifyAndSplitPass());
  pm.addNestedPass<func::FuncOp>(pto::createCVInsertBridgePass());
}
// 下游 pass 始终运行（不要用 if(!enableCVSeparation) 包裹）
pm.addNestedPass<func::FuncOp>(pto::createLoweringSyncToPipePass());
// ... 后续 pass
```

**Commit**
```bash
git commit -m "feat: integrate CV separation passes into pipeline"
```

---

### Task 9: 补充测试

**Files:**
- Create: `test/basic/cv_pure_cube.mlir` — 完整 Cube 路径 (GM→MAT→LEFT/RIGHT→matmul)
- Create: `test/basic/cv_tmov_classify.mlir` — TMOV 地址空间分类
- Create: `test/basic/cv_no_cross_dep.mlir` — 无跨域依赖
- Create: `test/basic/cv_existing_sections.mlir` — 保留已有 section

所有测试 CHECK 模式匹配 C++ 输出中的 `__DAV_CUBE__` / `__DAV_VEC__` 预处理守卫和指令名（TLOAD, TSTORE, TMOV, TMATMUL）。

**Commit**
```bash
git commit -m "test: add comprehensive CV separation tests"
```

---

## 实现注意事项

### 关键参考文件
- **Pass 模式**：`lib/PTO/Transforms/PTOInsertCVMov.cpp`（PassWrapper 风格）
- **数据搬运**：`lib/PTO/Transforms/InsertLoadStoreForMixCV.cpp`（TLoad/TStore 用法：`TStoreOp(loc, TypeRange{}, src, dst)`）
- **同步 Op**：`include/PTO/IR/PTOOps.td` — SyncSetOp: `PipeAttr + I32Attr`
- **Section Op**：`include/PTO/IR/PTOOps.td` — `SingleBlock, NoTerminator, SizedRegion<1>`
- **地址空间**：`include/PTO/IR/PTOAttrs.td` — Zero=0, GM=1, MAT=2, LEFT=3, RIGHT=4, ACC=5, VEC=6, BIAS=7, SCALING=8
- **Pipe 枚举**：`include/PTO/Transforms/SyncCommon.h` — `pto::PIPE::PIPE_MTE2/MTE3/FIX/M/V`

### 已知坑点
1. **SectionOp 空 region**：`builder.create<SectionCubeOp>()` 创建 0 个 block 的 region，访问 `.front()` 会 SEGFAULT，必须先 `region.emplaceBlock()`
2. **Pass 注册**：不要调用 `pto::registerPTOPasses()`，部分 pass 缺少 `getArgument()` 实现会导致 LLVM ERROR
3. **PipeAttr 构造**：使用 `pto::PIPE::PIPE_MTE3`（不是 `pto::PipelineType::PIPE_MTE3`）
4. **Pipeline 集成**：不要用 `if(!enableCVSeparation)` 包裹下游 pass，CV pass 应该与下游 pass 串联执行
