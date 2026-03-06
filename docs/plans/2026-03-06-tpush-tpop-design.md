# TPUSH/TPOP PTOAS Compiler Support Design

## 1. Overview

This document specifies the PTOAS compiler-side design for supporting `TPUSH`/`TPOP` instructions — a ring-buffer-based, intra-cluster data communication mechanism between Cube and Vector cores.

TPUSH/TPOP replaces the current GM bridge (TStore→GM→TLoad) in `CVInsertBridge` as the primary cross-section data transfer channel. It supports both A2/A3 (ring buffer in GM) and A5 (ring buffer in consumer's on-chip SRAM, zero-copy) platforms through platform-agnostic IR ops that lower differently per `--pto-arch`.

### References

- ISA-level design: `HL_ptoisa_newfeature20260306_TPUSH_TPOP.md`
- CV separation design: `docs/plans/2026-03-05-cv-separation-design-v2.md`

---

## 2. New Op Definitions (7 Ops)

All ops follow the existing PTOAS conventions: side-effect model (no SSA-threaded state), `OpPipeInterface` for sync insertion, `MemoryEffectsOpInterface` for optimization safety.

### 2.1 Memory Declaration Op

#### `pto.reserve_buffer`

Declares a reserved SRAM region in a consumer InCore function's local memory for ring buffer slots. Defined at **function level** (outside any section), so both `section_cube` and `section_vector` can reference its result.

```tablegen
def ReserveBufferOp : PTO_Op<"reserve_buffer", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Declare a reserved SRAM region for ring buffer slots";
    let arguments = (ins
        StrAttr:$name,                          // buffer identifier
        I32Attr:$size,                          // total bytes = SLOT_NUM * SLOT_SIZE
        OptionalAttr<I32Attr>:$base,            // base address (absent = compiler auto-assigns)
        PTO_AddressSpaceAttr:$memory_space      // VEC (Vector UB) or MAT (Cube L1)
    );
    let results = (outs I32:$result);           // resolved base address (compile-time constant)
    let hasVerifier = 1;
}
```

**Verifier rules:**
- Must be at function level (not inside any `SectionCubeOp` or `SectionVectorOp`)
- `size` > 0
- `memory_space` must be `VEC` or `MAT`
- If `base` is specified, must satisfy hardware alignment requirements (e.g., 32-byte aligned)

**Semantics:**
- On A5: reserves a segment in consumer's SRAM. `PTOPlanMemory` resolves the address.
- On A2A3: the op is still emitted for uniformity but the resolved address is not used at runtime (ring buffer is in GM via `gm_slot_buffer`). The lowering pass may fold it to a constant 0.

### 2.2 Initialization Ops

#### `pto.aic_initialize_pipe`

Called once at Cube kernel startup. Initializes the ring buffer pipe(s) for the specified direction(s). Defined at **function level**.

```tablegen
def AicInitializePipeOp : PTO_Op<"aic_initialize_pipe", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Initialize ring buffer pipe(s) on Cube core";
    let arguments = (ins
        I8Attr:$dir_mask,                       // 0b01=C2V, 0b10=V2C, 0b11=bidirectional
        I32Attr:$slot_size,                     // bytes per slot (= Tile size)
        PTODpsType:$gm_slot_buffer,             // GM buffer (A2A3 active; A5 nullptr)
        I32:$c2v_consumer_buf,                  // consumer SRAM base for C2V direction
        I32:$v2c_consumer_buf                   // consumer SRAM base for V2C direction
    );
    let results = (outs);
    let hasVerifier = 1;
}
```

#### `pto.aiv_initialize_pipe`

Called once at Vector kernel startup. Same signature.

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

**Shared verifier rules:**
- Must be at function level (not inside any section)
- `dir_mask` must be `1` (C2V), `2` (V2C), or `3` (bidirectional)
- `slot_size` > 0 and hardware-aligned
- `c2v_consumer_buf` and `v2c_consumer_buf` typically come from `reserve_buffer` results

### 2.3 Data Transfer Ops

#### `pto.tpush_to_aiv` (Cube producer, C2V direction)

```tablegen
def TPushToAivOp : PTO_TOp<"tpush_to_aiv", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Push tile from Cube to buddy Vector via ring buffer";
    let arguments = (ins
        PTODpsType:$tile,                       // source tile data
        I8Attr:$aiv_idx                         // target buddy Vector core index (0 or 1)
    );
    let results = (outs);
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE1; }
    }];
}
```

#### `pto.tpush_to_aic` (Vector producer, V2C direction)

```tablegen
def TPushToAicOp : PTO_TOp<"tpush_to_aic", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Push tile from Vector to buddy Cube via ring buffer";
    let arguments = (ins
        PTODpsType:$tile,                       // source tile data
        I8Attr:$aiv_idx                         // this Vector core's own index (0 or 1)
    );
    let results = (outs);
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE2; }
    }];
}
```

#### `pto.tpop_from_aic` (Vector consumer, C2V direction)

```tablegen
def TPopFromAicOp : PTO_TOp<"tpop_from_aic", [
    PTO_DpsInitOpInterface,
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Pop tile that Cube pushed, into destination tile buffer";
    let arguments = (ins
        PTODpsType:$tile,                       // destination tile buffer (DPS init)
        I8Attr:$aiv_idx                         // this Vector core's own index (0 or 1)
    );
    let results = (outs);
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE1; }
        ::mlir::MutableOperandRange getDpsInitsMutable() { return getTileMutable(); }
    }];
}
```

#### `pto.tpop_from_aiv` (Cube consumer, V2C direction)

```tablegen
def TPopFromAivOp : PTO_TOp<"tpop_from_aiv", [
    PTO_DpsInitOpInterface,
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
    let summary = "Pop tile that Vector pushed, into destination tile buffer";
    let arguments = (ins
        PTODpsType:$tile,                       // destination tile buffer (DPS init)
        I8Attr:$aiv_idx                         // source buddy Vector core index (0 or 1)
    );
    let results = (outs);
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE2; }
        ::mlir::MutableOperandRange getDpsInitsMutable() { return getTileMutable(); }
    }];
}
```

**Shared verifier rules for data transfer ops:**
- `aiv_idx` must be 0 or 1
- `tpush_to_aiv` / `tpop_from_aiv` must be inside `SectionCubeOp`
- `tpush_to_aic` / `tpop_from_aic` must be inside `SectionVectorOp`
- `tile` must be of `TileBufType`

### 2.4 Op Summary

| Op | Location | Core | Role | Pipe | DPS |
|---|---|---|---|---|---|
| `pto.reserve_buffer` | Function level | — | Memory declaration | None | No (returns i32) |
| `pto.aic_initialize_pipe` | Function level | Cube | Setup | None | No |
| `pto.aiv_initialize_pipe` | Function level | Vector | Setup | None | No |
| `pto.tpush_to_aiv` | SectionCubeOp | Cube | Producer (C2V) | PIPE_MTE1 | No (reads tile) |
| `pto.tpush_to_aic` | SectionVectorOp | Vector | Producer (V2C) | PIPE_MTE2 | No (reads tile) |
| `pto.tpop_from_aic` | SectionVectorOp | Vector | Consumer (C2V) | PIPE_MTE1 | Yes (writes tile) |
| `pto.tpop_from_aiv` | SectionCubeOp | Cube | Consumer (V2C) | PIPE_MTE2 | Yes (writes tile) |

> **Note:** The PIPE assignments (MTE1/MTE2) are initial recommendations. Actual pipe mapping should be confirmed against Da Vinci hardware MTE channel constraints.

---

## 3. CV Separation Integration

### 3.1 Current CV Separation Pipeline (merged)

```
Mixed InCore function (Cube + Vector ops interleaved)
        |
        v
  CVClassifyAndSplit
  -- Classify ops by hardware domain
  -- Wrap into SectionCubeOp / SectionVectorOp regions
        |
        v
  CVInsertBridge
  -- Detect cross-domain data dependencies
  -- Insert GM workspace bridge: TStore(src -> GM) + TLoad(GM -> dst)
        |
        v
  PTOInsertSync -> PTOPlanMemory -> PTOToEmitC
```

### 3.2 New Pipeline with TPUSH/TPOP

```
Mixed InCore function
        |
        v
  CVClassifyAndSplit                 <-- unchanged
        |
        v
  CVInsertBridge                     <-- EXTENDED: emit tpush/tpop instead of GM bridge
  -- Detect cross-domain data dependencies
  -- For each dependency edge: emit tpush/tpop pair
  -- Emit reserve_buffer at function level (for A5 consumer SRAM)
  -- Emit initialize_pipe at function level
        |
        v
  PTOInsertSync                      <-- unchanged (auto-adapts via OpPipeInterface)
        |
        v
  PTOPlanMemory                      <-- EXTENDED: handle reserve_buffer
        |
        v
  PTOToEmitC                         <-- EXTENDED: 7 new lowering patterns
```

### 3.3 CVInsertBridge Extension

#### Current behavior (GM bridge)

For each cross-domain data dependency `%v` (defined in SectionA, used in SectionB):
1. Insert `pto.tstore %v -> %gm_workspace` in the producer section
2. Insert `pto.tload %gm_workspace -> %v_copy` in the consumer section
3. Replace use of `%v` in consumer section with `%v_copy`

#### New behavior (TPUSH/TPOP ring buffer)

For each cross-domain data dependency `%v`:

**Step 1 — Emit function-level setup (once per function, before all sections):**

```mlir
// Ring buffer declaration (for A5 consumer SRAM reservation)
%buf_c2v = pto.reserve_buffer {
    name = "c2v_slot_buffer",
    size = SLOT_NUM * SLOT_SIZE,
    memory_space = VEC                  // or MAT, depending on consumer core type
}

// Initialization for both cores
pto.aic_initialize_pipe {dir_mask = 1, slot_size = SLOT_SIZE}
    (%gm_slot_buf, %buf_c2v, %cst0)
pto.aiv_initialize_pipe {dir_mask = 1, slot_size = SLOT_SIZE}
    (%gm_slot_buf, %buf_c2v, %cst0)
```

For bidirectional communication (both C2V and V2C dependencies exist):

```mlir
%buf_c2v = pto.reserve_buffer {name = "c2v", size = ..., memory_space = VEC}
%buf_v2c = pto.reserve_buffer {name = "v2c", size = ..., memory_space = MAT}

pto.aic_initialize_pipe {dir_mask = 3, slot_size = SLOT_SIZE}
    (%gm_slot_buf, %buf_c2v, %buf_v2c)
pto.aiv_initialize_pipe {dir_mask = 3, slot_size = SLOT_SIZE}
    (%gm_slot_buf, %buf_c2v, %buf_v2c)
```

**Step 2 — Emit tpush in the producer section:**

```mlir
pto.section_cube {
    ...
    pto.tpush_to_aiv(%v) {aiv_idx = 0}    // push the cross-domain value
}
```

**Step 3 — Emit tpop in the consumer section:**

```mlir
pto.section_vector {
    %v_recv = pto.alloc_tile ...
    pto.tpop_from_aic(%v_recv) {aiv_idx = 0}
    // replace all uses of %v with %v_recv
    ...
}
```

#### SLOT_SIZE and SLOT_NUM determination

`CVInsertBridge` computes these from the cross-domain tile:

- `SLOT_SIZE` = byte size of the transferred tile (`TileBufType` shape * element size)
- `SLOT_NUM` = 8 (unidirectional) or 4 (bidirectional), determined by `dir_mask`

#### Multiple cross-domain dependencies

If multiple tile values cross the same domain boundary in the same direction, they are serialized through the same ring buffer channel. The producer pushes them in order; the consumer pops them in the same order (FIFO guarantee).

If tiles cross in **both** directions (Cube->Vector and Vector->Cube), bidirectional mode is used (`dir_mask = 3`, `SLOT_NUM = 4`).

### 3.4 PTOPlanMemory Extension

`PTOPlanMemory` must handle `reserve_buffer` ops:

```
For each reserve_buffer op in the function:

    if base attribute is present (manually specified):
        validate [base, base + size) does not overlap with prior allocations
        mark region as occupied
        resolve op result to the literal base value

    if base attribute is absent (auto mode):
        add to pending allocation list

Allocate all normal tiles and temporaries, skipping occupied regions.

For each pending auto reserve_buffer:
    find a free region of the required size in the appropriate SRAM
    assign address, mark as occupied
    resolve op result to the assigned address (constant fold)
```

**Address allocator contract:** The segment `[BASE, BASE + SIZE)` is reserved. No other tile, temporary, or spill allocation may overlap with it.

### 3.5 PTOInsertSync — No Changes Required

The `PTOInsertSync` pass already works through `OpPipeInterface`:

- `tpush_to_aiv` declares `PIPE_MTE1` -> sync insertion handles MTE1 boundaries automatically
- `tpop_from_aic` declares `PIPE_MTE1` -> same
- `tpush_to_aic` declares `PIPE_MTE2` -> sync insertion handles MTE2 boundaries automatically
- `tpop_from_aiv` declares `PIPE_MTE2` -> same

Cross-core synchronization (the flag SET/WAIT between Cube and Vector) is **internal** to the tpush/tpop semantics and emitted during EmitC lowering. `PTOInsertSync` only handles intra-core pipe synchronization.

### 3.6 PTOToEmitC Lowering

7 new conversion patterns:

| Pattern | Input Op | EmitC Output |
|---|---|---|
| `ReserveBufferToEmitC` | `pto.reserve_buffer` | Eliminated (resolved to `emitc::ConstantOp` by PlanMemory) |
| `AicInitPipeToEmitC` | `pto.aic_initialize_pipe` | `emitc::CallOpaqueOp("aic_initialize_pipe", dir_mask, slot_size, gm_buf, c2v_buf, v2c_buf)` |
| `AivInitPipeToEmitC` | `pto.aiv_initialize_pipe` | `emitc::CallOpaqueOp("aiv_initialize_pipe", dir_mask, slot_size, gm_buf, c2v_buf, v2c_buf)` |
| `TPushToAivToEmitC` | `pto.tpush_to_aiv` | `emitc::CallOpaqueOp("tpush_to_aiv", tile, aiv_idx)` |
| `TPushToAicToEmitC` | `pto.tpush_to_aic` | `emitc::CallOpaqueOp("tpush_to_aic", tile, aiv_idx)` |
| `TPopFromAicToEmitC` | `pto.tpop_from_aic` | `emitc::CallOpaqueOp("tpop_from_aic", tile, aiv_idx)` |
| `TPopFromAivToEmitC` | `pto.tpop_from_aiv` | `emitc::CallOpaqueOp("tpop_from_aiv", tile, aiv_idx)` |

The EmitC output calls map directly to pto-isa C++ library functions. Platform-specific behavior (GM DMA vs SRAM zero-copy) is handled inside the pto-isa library at runtime via `PLATFORM_ID`.

---

## 4. End-to-End Example

### 4.1 Input: Mixed InCore kernel (user-written)

```python
@pl.incore
def matmul_relu_kernel(gm_a, gm_b, gm_c):
    tile_a = pl.tload(gm_a)
    tile_c = pl.tmatmul(tile_a, tile_b)    # Cube op
    tile_out = pl.trelu(tile_c)            # Vector op -- cross-domain dependency
    pl.tstore(tile_out, gm_c)
```

### 4.2 After CVClassifyAndSplit

```mlir
func @matmul_relu_kernel(%gm_a, %gm_b, %gm_c, %gm_slot_buf) {
    pto.section_cube {
        %a = pto.alloc_tile ...
        pto.tload(%gm_a, %a)
        %c = pto.alloc_tile ...
        pto.tmatmul(%a, %b, %c)
        // %c is used by vector section -- cross-domain dependency
    }
    pto.section_vector {
        %out = pto.alloc_tile ...
        pto.trelu(%c, %out)              // %c comes from cube section
        pto.tstore(%out, %gm_c)
    }
}
```

### 4.3 After CVInsertBridge (extended with TPUSH/TPOP)

```mlir
func @matmul_relu_kernel(%gm_a, %gm_b, %gm_c, %gm_slot_buf) {
    // --- Function level: ring buffer setup ---
    %cst0 = arith.constant 0 : i32
    %buf = pto.reserve_buffer {
        name = "c2v_slot_buffer",
        size = 32768,                       // 8 slots * 4096 bytes
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

### 4.4 After PTOPlanMemory

```mlir
// reserve_buffer resolved: %buf folded to constant 0x1000
// alloc_tile addresses allocated outside [0x1000, 0x9000)
// All tile addresses are resolved constants
```

### 4.5 After PTOToEmitC (final C++ output)

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

### 4.6 Bidirectional Example (Cube->Vector + Vector->Cube)

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

## 5. Files to Modify

| Layer | File | Change |
|---|---|---|
| **ODS** | `include/PTO/IR/PTOOps.td` | Add 7 new op definitions |
| **C++ IR** | `lib/PTO/IR/PTO.cpp` | Add verifier implementations for 7 ops |
| **CV Bridge** | `lib/PTO/Transforms/CVInsertBridge.cpp` | Extend to emit tpush/tpop + reserve_buffer + initialize_pipe |
| **PlanMemory** | `lib/PTO/Transforms/PTOPlanMemory.cpp` | Handle reserve_buffer: reservation, auto-allocation, constant folding |
| **EmitC** | `lib/PTO/Transforms/PTOToEmitC.cpp` | Add 7 conversion patterns |
| **Python** | `python/pto/dialects/PTOOps.td` | Auto-generated (wrapper includes PTOOps.td) |
| **Tests** | `test/basic/tpush_tpop_*.mlir` | Regression tests for ops, verifiers, and lowering |

---

## 6. Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| State model | Side-effect (no SSA threading) | Consistent with existing set_flag/wait_flag/get_buf/rls_buf |
| Platform handling | Single op set, arch-aware lowering | Matches existing pattern (TLoadOp etc.), avoids op explosion |
| Buffer type | No new type; reserve_buffer returns i32 | Minimal IR complexity; ring buffer state lives in lowered C++ |
| Op location | reserve_buffer + initialize_pipe at function level; tpush/tpop inside sections | MLIR SSA scoping: function-level values are visible to nested regions |
| import_peer_buffer | Removed | Unnecessary — both sections are in same function, can share SSA values |
| Direction encoding | In opcode (4 distinct ops) | Matches ISA design; enables static verifier checks per core type |
| CV bridge integration | Extend CVInsertBridge | Reuses existing cross-domain dependency detection; tpush/tpop replaces GM bridge |
| InsertSync | No changes | OpPipeInterface auto-adapts; cross-core sync is internal to tpush/tpop semantics |
