# 1. Pipeline Synchronization

> **Category:** Synchronization primitives for coordinating pipeline execution
> **Pipelines:** MTE2 (GM→UB), PIPE_V (Vector), MTE3 (UB→GM)

VPTO operates on the Ascend 950's **Decoupled Access-Execute** architecture. The MTE and Vector pipelines run asynchronously, requiring explicit synchronization to prevent data hazards.

---

## Intra-Core Pipeline Sync

These ops coordinate data flow between pipelines within a single vector core.

### `pto.set_flag`

- **syntax:** `pto.set_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- **CCE:** `__builtin_cce_set_flag`
- **semantics:** Signal event from source pipe to destination pipe.

```c
set_flag(src_pipe, dst_pipe, event_id);
```

**Example:** After MTE2 completes GM→UB transfer, signal Vector pipe:
```mlir
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

---

### `pto.wait_flag`

- **syntax:** `pto.wait_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- **CCE:** `__builtin_cce_wait_flag`
- **semantics:** Block destination pipe until source pipe signals event.

```c
wait_flag(src_pipe, dst_pipe, event_id);
```

**Example:** Vector pipe waits for MTE2 data to arrive:
```mlir
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

---

### `pto.pipe_barrier`

- **syntax:** `pto.pipe_barrier "PIPE_*"`
- **CCE:** `__builtin_cce_pipe_barrier`
- **semantics:** Drain all pending ops in the specified pipe.

```c
pipe_barrier(pipe);
```

**Pipe identifiers:** `PIPE_MTE2`, `PIPE_V`, `PIPE_MTE3`

---

### `pto.get_buf`

- **syntax:** `pto.get_buf "PIPE_*", %buf_id, %mode : i64, i64`
- **CCE:** `__builtin_cce_get_buf`
- **semantics:** Acquire buffer slot for inter-pipeline double-buffering coordination.

```c
get_buf(pipe, buf_id, mode);
```

---

### `pto.rls_buf`

- **syntax:** `pto.rls_buf "PIPE_*", %buf_id, %mode : i64, i64`
- **CCE:** `__builtin_cce_rls_buf`
- **semantics:** Release buffer slot to allow other pipeline to proceed.

```c
rls_buf(pipe, buf_id, mode);
```

---

### `pto.mem_bar`

- **syntax:** `pto.mem_bar "BARRIER_TYPE"`
- **CCE:** `__builtin_cce_pipe_barrier` (PIPE_V context)
- **semantics:** Intra-vector-pipe memory fence within `__VEC_SCOPE__`. Required when UB addresses alias between vector load/store operations.

```c
mem_bar(barrier_type);
```

**Barrier types:**

| Type | Semantics |
|------|-----------|
| `VV_ALL` | All prior vector ops complete before subsequent |
| `VST_VLD` | All prior vector stores visible before subsequent loads |
| `VLD_VST` | All prior vector loads complete before subsequent stores |

**Example:** Ensure stores are visible before loads to same UB region:
```mlir
pto.vsts %v0, %ub[%c0] : !pto.vreg<64xf32>, !llvm.ptr<6>
pto.mem_bar "VST_VLD"
%v1 = pto.vlds %ub[%c0] : !llvm.ptr<6> -> !pto.vreg<64xf32>
```

---

## Inter-Core Sync

These ops coordinate execution across multiple vector cores or AI cores.

### `pto.set_cross_core`

- **syntax:** `pto.set_cross_core %core_id, %event_id : i64, i64`
- **CCE:** `__builtin_cce_set_cross_core`
- **semantics:** Signal event to another core.

```c
set_cross_core(target_core, event_id);
```

---

### `pto.wait_flag_dev`

- **syntax:** `pto.wait_flag_dev %core_id, %event_id : i64, i64`
- **CCE:** `__builtin_cce_wait_flag_dev`
- **semantics:** Wait for event from another core.

```c
wait_flag_dev(source_core, event_id);
```

---

### `pto.set_intra_block`

- **syntax:** `pto.set_intra_block %block_id, %event_id : i64, i64`
- **CCE:** `__builtin_cce_set_intra_block`
- **semantics:** Signal event within a block of cores.

```c
set_intra_block(block_id, event_id);
```

---

### `pto.wait_intra_block`

- **syntax:** `pto.wait_intra_block %block_id, %event_id : i64, i64`
- **CCE:** `__builtin_cce_wait_intra_block`
- **semantics:** Wait for event from any core in the block.

```c
wait_intra_block(block_id, event_id);
```

---

## Typical Sync Pattern

```mlir
// 1. Configure and start GM→UB DMA
pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr, ...

// 2. Signal DMA complete
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

// 3. Vector pipe waits for data
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

// 4. Vector computation
scf.for %dummy = %c0 to %c1 step %c1 {
  %v = pto.vlds %ub_ptr[%lane] : !llvm.ptr<6> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane] : !pto.vreg<64xf32>, !llvm.ptr<6>
} {llvm.loop.aivector_scope}

// 5. Signal compute complete
pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]

// 6. MTE3 waits then stores to GM
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.copy_ubuf_to_gm %ub_out, %gm_out, ...
```
