# CV Separation Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix CV separation passes based on hardware data path feedback (信息补充.md).

**Architecture:** Minimal targeted fixes to existing CVClassifyAndSplit and CVInsertBridge passes — fix address space classification, TMovOp handling, sync pipe selection, and pipeline integration.

**Tech Stack:** C++ / MLIR / PTO dialect / FileCheck lit tests

---

### Task 1: Fix classifyOp() — add MAT, BIAS, SCALING to cubeSpaces

**Files:**
- Modify: `lib/PTO/Transforms/CVClassifyAndSplit.cpp:148-150`

**Step 1: Update cubeSpaces array**

Change lines 148-150 from:
```cpp
    static const pto::AddressSpace cubeSpaces[] = {
        pto::AddressSpace::LEFT, pto::AddressSpace::RIGHT,
        pto::AddressSpace::ACC};
```
To:
```cpp
    static const pto::AddressSpace cubeSpaces[] = {
        pto::AddressSpace::LEFT, pto::AddressSpace::RIGHT,
        pto::AddressSpace::ACC, pto::AddressSpace::MAT,
        pto::AddressSpace::BIAS, pto::AddressSpace::SCALING};
```

**Step 2: Commit**

```bash
git add lib/PTO/Transforms/CVClassifyAndSplit.cpp
git commit -m "fix: add MAT, BIAS, SCALING to cube address spaces"
```

---

### Task 2: Fix classifyOp() — remove TMovOp from VECTOR op list

**Files:**
- Modify: `lib/PTO/Transforms/CVClassifyAndSplit.cpp:144`

**Step 1: Remove TMovOp from Vector op type list**

Change line 144 from:
```cpp
    if (isa<AddFOp, AddFDpsOp, TransOp, TTransOp, TMovOp, MovOp>(op))
```
To:
```cpp
    if (isa<AddFOp, AddFDpsOp, TransOp, TTransOp, MovOp>(op))
```

TMovOp will now fall through to address-space matching:
- `tmov(MAT → LEFT)` → operands touch MAT/LEFT → CUBE
- `tmov(VEC → VEC)` → operands touch VEC → VECTOR

**Step 2: Commit**

```bash
git add lib/PTO/Transforms/CVClassifyAndSplit.cpp
git commit -m "fix: classify TMovOp by address space, not op type"
```

---

### Task 3: Fix insertBridge() — select sync pipe by section type

**Files:**
- Modify: `lib/PTO/Transforms/CVInsertBridge.cpp:93-95`

**Step 1: Choose pipe based on producer section type**

Change lines 93-95 from:
```cpp
    // sync.set on MTE3 pipe (store pipe)
    auto pipeAttr =
        PipeAttr::get(builder.getContext(), pto::PIPE::PIPE_MTE3);
```
To:
```cpp
    // sync.set pipe: Cube section (ACC→GM) uses PIPE_FIX, Vector section (UB→GM) uses MTE3
    auto storePipe = isa<SectionCubeOp>(bp.producerSection)
                         ? pto::PIPE::PIPE_FIX
                         : pto::PIPE::PIPE_MTE3;
    auto pipeAttr = PipeAttr::get(builder.getContext(), storePipe);
```

**Step 2: Commit**

```bash
git add lib/PTO/Transforms/CVInsertBridge.cpp
git commit -m "fix: use PIPE_FIX for cube→GM bridge, MTE3 for vector→GM"
```

---

### Task 4: Fix ptoas.cpp — don't skip downstream passes in CV mode

**Files:**
- Modify: `tools/ptoas/ptoas.cpp:550-596`

**Step 1: Restructure pipeline to insert CV passes without skipping downstream**

Replace the current pipeline logic (lines 550-596):

```cpp
  // CV Separation: classify ops and split into sections, then insert bridges
  if (enableCVSeparation) {
    pm.addNestedPass<mlir::func::FuncOp>(pto::createCVClassifyAndSplitPass());
    pm.addNestedPass<mlir::func::FuncOp>(pto::createCVInsertBridgePass());
  }

  if (!enableCVSeparation) {
    // Full pipeline (skip when running CV separation only)
    pm.addNestedPass<mlir::func::FuncOp>(pto::createLoweringSyncToPipePass());
    ...
  }
```

With:

```cpp
  // CV Separation: classify ops and split into sections, then insert bridges
  if (enableCVSeparation) {
    pm.addNestedPass<mlir::func::FuncOp>(pto::createCVClassifyAndSplitPass());
    pm.addNestedPass<mlir::func::FuncOp>(pto::createCVInsertBridgePass());
  }

  // Downstream pipeline (always runs)
  pm.addNestedPass<mlir::func::FuncOp>(pto::createLoweringSyncToPipePass());

  pm.addPass(pto::createPTOViewToMemrefPass());
  if (!disableInferLayout)
    pm.addNestedPass<mlir::func::FuncOp>(pto::createInferPTOLayoutPass());

  PlanMemoryOptions planMemoryOption;
  planMemoryOption.memMode = MemPlanMode::GLOBAL_WORKSPACE_PLAN;
  planMemoryOption.enableGlobalReuse = false;
  planMemoryOption.enablePrintMemoryAllocatedSize = false;
  pm.addPass(pto::createPlanMemoryPass());

  if (enableInsertSync) {
    pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOInsertSyncPass());
  }

  pm.addPass(createCSEPass());
  pm.addPass(pto::createEmitPTOManualPass());
  pm.addPass(emitc::createFormExpressionsPass());
  pm.addPass(mlir::createCSEPass());
```

Also remove the early-exit CV dump block (lines 591-596):
```cpp
  // DELETE these lines:
  // CV separation mode: dump MLIR IR and exit
  if (enableCVSeparation) {
    module->print(outputFile.os());
    outputFile.keep();
    return 0;
  }
```

**Step 2: Commit**

```bash
git add tools/ptoas/ptoas.cpp
git commit -m "fix: CV separation passes integrate into pipeline without skipping downstream"
```

---

### Task 5: Update test — cv_pure_cube.mlir with realistic Cube data path

**Files:**
- Modify: `test/basic/cv_pure_cube.mlir`

**Step 1: Rewrite test with GM → MAT → LEFT/RIGHT → matmul → ACC path**

```mlir
// RUN: ptoas --enable-cv-separation %s | FileCheck %s

// Test: complete cube data path — tload GM→MAT, tmov MAT→LEFT/RIGHT, tmatmul, all in section.cube.

module {
  func.func @cube_full_path(
      %gm_a: memref<16x256xf16, #pto.address_space<gm>>,
      %gm_b: memref<256x16xf16, #pto.address_space<gm>>,
      %mat_a: memref<16x256xf16, #pto.address_space<mat>>,
      %mat_b: memref<256x16xf16, #pto.address_space<mat>>,
      %left: memref<16x256xf16, #pto.address_space<left>>,
      %right: memref<256x16xf16, #pto.address_space<right>>,
      %acc: memref<16x16xf32, #pto.address_space<acc>>) {
    pto.tload ins(%gm_a : memref<16x256xf16, #pto.address_space<gm>>)
              outs(%mat_a : memref<16x256xf16, #pto.address_space<mat>>)
    pto.tload ins(%gm_b : memref<256x16xf16, #pto.address_space<gm>>)
              outs(%mat_b : memref<256x16xf16, #pto.address_space<mat>>)
    pto.tmov ins(%mat_a : memref<16x256xf16, #pto.address_space<mat>>)
             outs(%left : memref<16x256xf16, #pto.address_space<left>>)
    pto.tmov ins(%mat_b : memref<256x16xf16, #pto.address_space<mat>>)
             outs(%right : memref<256x16xf16, #pto.address_space<right>>)
    pto.tmatmul ins(%left, %right : memref<16x256xf16, #pto.address_space<left>>, memref<256x16xf16, #pto.address_space<right>>) outs(%acc : memref<16x16xf32, #pto.address_space<acc>>)
    return
  }
}

// CHECK: pto.section.cube
// CHECK:   pto.tload
// CHECK:   pto.tload
// CHECK:   pto.tmov
// CHECK:   pto.tmov
// CHECK:   pto.tmatmul
// CHECK-NOT: pto.section.vector
```

**Step 2: Commit**

```bash
git add test/basic/cv_pure_cube.mlir
git commit -m "test: update cv_pure_cube with realistic cube data path (GM→MAT→LEFT/RIGHT→matmul)"
```

---

### Task 6: Add test — cv_tmov_classify.mlir for TMOV address-space classification

**Files:**
- Create: `test/basic/cv_tmov_classify.mlir`

**Step 1: Write test that verifies TMOV is classified by address space**

```mlir
// RUN: ptoas --enable-cv-separation %s | FileCheck %s

// Test: TMOV classified by address space — MAT→LEFT is CUBE, VEC→VEC is VECTOR.

module {
  func.func @tmov_classify(
      %mat_buf: memref<16x256xf16, #pto.address_space<mat>>,
      %left: memref<16x256xf16, #pto.address_space<left>>,
      %ub_src: memref<16x256xf16, #pto.address_space<vec>>,
      %ub_dst: memref<16x256xf16, #pto.address_space<vec>>) {
    // This tmov touches MAT and LEFT → should be CUBE
    pto.tmov ins(%mat_buf : memref<16x256xf16, #pto.address_space<mat>>)
             outs(%left : memref<16x256xf16, #pto.address_space<left>>)
    // This tmov touches only VEC → should be VECTOR
    pto.tmov ins(%ub_src : memref<16x256xf16, #pto.address_space<vec>>)
             outs(%ub_dst : memref<16x256xf16, #pto.address_space<vec>>)
    return
  }
}

// CHECK: pto.section.cube
// CHECK:   pto.tmov
// CHECK: pto.section.vector
// CHECK:   pto.tmov
```

**Step 2: Commit**

```bash
git add test/basic/cv_tmov_classify.mlir
git commit -m "test: add TMOV address-space classification test"
```

---

### Task 7: Update remaining tests for correct data paths

**Files:**
- Modify: `test/basic/cv_no_cross_dep.mlir`
- Modify: `test/basic/cv_split_loop.mlir`
- Modify: `test/basic/cv_bridge_cube_to_vec.mlir`
- Modify: `test/basic/cv_existing_sections.mlir`

**Step 1: Update cv_no_cross_dep.mlir — use MAT-based cube ops**

```mlir
// RUN: ptoas --enable-cv-separation %s | FileCheck %s

// Test: no cross-domain dependency — no bridge ops inserted.

module {
  func.func @no_cross_dep(
      %gm_a: memref<16x256xf16, #pto.address_space<gm>>,
      %mat_a: memref<16x256xf16, #pto.address_space<mat>>,
      %left: memref<16x256xf16, #pto.address_space<left>>,
      %right: memref<256x16xf16, #pto.address_space<right>>,
      %acc: memref<16x16xf32, #pto.address_space<acc>>,
      %ub_buf: memref<16x16xf32, #pto.address_space<vec>>,
      %gm_out: memref<16x16xf32, #pto.address_space<gm>>,
      %workspace: memref<16x16xf32, #pto.address_space<gm>>) {
    pto.tload ins(%gm_a : memref<16x256xf16, #pto.address_space<gm>>)
              outs(%mat_a : memref<16x256xf16, #pto.address_space<mat>>)
    pto.tmov ins(%mat_a : memref<16x256xf16, #pto.address_space<mat>>)
             outs(%left : memref<16x256xf16, #pto.address_space<left>>)
    pto.tmatmul ins(%left, %right : memref<16x256xf16, #pto.address_space<left>>, memref<256x16xf16, #pto.address_space<right>>) outs(%acc : memref<16x16xf32, #pto.address_space<acc>>)
    pto.tstore ins(%ub_buf : memref<16x16xf32, #pto.address_space<vec>>) outs(%gm_out : memref<16x16xf32, #pto.address_space<gm>>)
    return
  }
}

// CHECK: pto.section.cube
// CHECK: pto.section.vector
// CHECK-NOT: pto.sync.set
// CHECK-NOT: pto.sync.wait
```

**Step 2: Update cv_split_loop.mlir — add MAT in cube path**

```mlir
// RUN: ptoas --enable-cv-separation %s | FileCheck %s

// Test: scf.for containing both cube and vector ops is split into
// two parallel loops, one in each section.

module {
  func.func @mixed_loop(
      %gm_a: memref<16x256xf16, #pto.address_space<gm>>,
      %mat_a: memref<16x256xf16, #pto.address_space<mat>>,
      %left: memref<16x256xf16, #pto.address_space<left>>,
      %right: memref<256x16xf16, #pto.address_space<right>>,
      %acc: memref<16x16xf32, #pto.address_space<acc>>,
      %ub_buf: memref<16x16xf32, #pto.address_space<vec>>,
      %gm_out: memref<16x16xf32, #pto.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      pto.tmov ins(%mat_a : memref<16x256xf16, #pto.address_space<mat>>)
               outs(%left : memref<16x256xf16, #pto.address_space<left>>)
      pto.tmatmul ins(%left, %right : memref<16x256xf16, #pto.address_space<left>>, memref<256x16xf16, #pto.address_space<right>>) outs(%acc : memref<16x16xf32, #pto.address_space<acc>>)
      pto.tstore ins(%ub_buf : memref<16x16xf32, #pto.address_space<vec>>) outs(%gm_out : memref<16x16xf32, #pto.address_space<gm>>)
    }
    return
  }
}

// CHECK: pto.section.cube
// CHECK:   scf.for
// CHECK:     pto.tmov
// CHECK:     pto.tmatmul
// CHECK: pto.section.vector
// CHECK:   scf.for
// CHECK:     pto.tstore
```

**Step 3: Update cv_bridge_cube_to_vec.mlir — keep as-is (DPS means no SSA bridges needed)**

No change needed — this test correctly verifies that DPS ops don't create cross-domain SSA dependencies.

**Step 4: Update cv_existing_sections.mlir — add MAT to cube section**

```mlir
// RUN: ptoas --enable-cv-separation %s | FileCheck %s

// Test: existing section ops are preserved as-is.

module {
  func.func @existing_sections(
      %mat_a: memref<16x256xf16, #pto.address_space<mat>>,
      %left: memref<16x256xf16, #pto.address_space<left>>,
      %right: memref<256x16xf16, #pto.address_space<right>>,
      %acc: memref<16x16xf32, #pto.address_space<acc>>,
      %ub_buf: memref<16x16xf32, #pto.address_space<vec>>,
      %gm_out: memref<16x16xf32, #pto.address_space<gm>>) {
    pto.section.cube {
      pto.tmov ins(%mat_a : memref<16x256xf16, #pto.address_space<mat>>)
               outs(%left : memref<16x256xf16, #pto.address_space<left>>)
      pto.tmatmul ins(%left, %right : memref<16x256xf16, #pto.address_space<left>>, memref<256x16xf16, #pto.address_space<right>>) outs(%acc : memref<16x16xf32, #pto.address_space<acc>>)
    }
    pto.tstore ins(%ub_buf : memref<16x16xf32, #pto.address_space<vec>>) outs(%gm_out : memref<16x16xf32, #pto.address_space<gm>>)
    return
  }
}

// CHECK: pto.section.cube
// CHECK:   pto.tmov
// CHECK:   pto.tmatmul
// CHECK: pto.section.vector
// CHECK:   pto.tstore
```

**Step 5: Commit**

```bash
git add test/basic/cv_no_cross_dep.mlir test/basic/cv_split_loop.mlir test/basic/cv_existing_sections.mlir
git commit -m "test: update test cases with realistic cube data paths (MAT, TMOV)"
```

---

### Task 8: Build and run all CV tests

**Step 1: Build**

```bash
cd /Users/kurrin/workspace/PTOAS/.worktrees/cv-separation/build
cmake --build . --target ptoas -j$(sysctl -n hw.ncpu)
```

Expected: Build succeeds with no errors.

**Step 2: Run all CV tests**

```bash
cd /Users/kurrin/workspace/PTOAS/.worktrees/cv-separation
for f in test/basic/cv_*.mlir; do
  echo "=== $f ==="
  build/bin/ptoas --enable-cv-separation "$f" | build/bin/FileCheck "$f"
  echo "exit: $?"
done
```

Expected: All tests pass (exit 0).

**Step 3: Run existing test to verify no regression**

```bash
build/bin/ptoas --enable-cv-separation test/basic/record_wait_event.mlir | build/bin/FileCheck test/basic/record_wait_event.mlir
```

Expected: Existing test still passes.

---
