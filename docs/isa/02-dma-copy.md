# 2. DMA Copy Programming

> **Category:** DMA transfer configuration and execution
> **Pipelines:** MTE2 (GM→UB), MTE3 (UB→GM)

DMA transfers move data between Global Memory (GM) and Unified Buffer (UB). The MTE engines operate asynchronously from the Vector core, requiring explicit sync (see [Pipeline Sync](01-pipeline-sync.md)).

---

## Loop Stride Configuration (GM→UB)

These ops configure the DMA engine for GM→UB transfers before calling `pto.copy_gm_to_ubuf`.

### `pto.set_loop2_stride_outtoub`

- **syntax:** `pto.set_loop2_stride_outtoub %src_stride, %dst_stride : i64, i64`
- **CCE:** `__builtin_cce_set_loop2_stride_outtoub`
- **semantics:** Configure outer loop stride for GM→UB DMA.

---

### `pto.set_loop1_stride_outtoub`

- **syntax:** `pto.set_loop1_stride_outtoub %src_stride, %dst_stride : i64, i64`
- **CCE:** `__builtin_cce_set_loop1_stride_outtoub`
- **semantics:** Configure inner loop stride for GM→UB DMA.

---

### `pto.set_loop_size_outtoub`

- **syntax:** `pto.set_loop_size_outtoub %loop1_count, %loop2_count : i64, i64`
- **CCE:** `__builtin_cce_set_loop_size_outtoub`
- **semantics:** Configure loop iteration counts for GM→UB DMA.

---

## Loop Stride Configuration (UB→GM)

These ops configure the DMA engine for UB→GM transfers before calling `pto.copy_ubuf_to_gm`.

### `pto.set_loop2_stride_ubtoout`

- **syntax:** `pto.set_loop2_stride_ubtoout %src_stride, %dst_stride : i64, i64`
- **CCE:** `__builtin_cce_set_loop2_stride_ubtoout`
- **semantics:** Configure outer loop stride for UB→GM DMA.

---

### `pto.set_loop1_stride_ubtoout`

- **syntax:** `pto.set_loop1_stride_ubtoout %src_stride, %dst_stride : i64, i64`
- **CCE:** `__builtin_cce_set_loop1_stride_ubtoout`
- **semantics:** Configure inner loop stride for UB→GM DMA.

---

### `pto.set_loop_size_ubtoout`

- **syntax:** `pto.set_loop_size_ubtoout %loop1_count, %loop2_count : i64, i64`
- **CCE:** `__builtin_cce_set_loop_size_ubtoout`
- **semantics:** Configure loop iteration counts for UB→GM DMA.

---

## DMA Transfer Execution

### `pto.copy_gm_to_ubuf`

- **syntax:**
```mlir
pto.copy_gm_to_ubuf %source, %dest, %valid_rows, %valid_cols, %sid, %n_burst, %len_burst,
    %left_padding, %right_padding, %l2_cache_ctl, %gm_stride, %ub_stride
    {layout = "LAYOUT", data_select_bit = true|false, ub_pad = true|false}
    : !llvm.ptr<1>, !llvm.ptr<6>, i64 x10
```
- **CCE:** `__builtin_cce_copy_gm_to_ubuf_align_v2`
- **semantics:** DMA transfer from Global Memory (AS=1) to Unified Buffer (AS=6).

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `%source` | GM source pointer (`!llvm.ptr<1>`) |
| `%dest` | UB destination pointer (`!llvm.ptr<6>`) |
| `%valid_rows` | Number of valid rows |
| `%valid_cols` | Number of valid columns (bytes) |
| `%sid` | Stream ID |
| `%n_burst` | Number of bursts |
| `%len_burst` | Length per burst (bytes) |
| `%left_padding` | Left padding (bytes) |
| `%right_padding` | Right padding (bytes) |
| `%l2_cache_ctl` | L2 cache control |
| `%gm_stride` | GM stride between rows |
| `%ub_stride` | UB stride between rows |

**Attributes:**

| Attribute | Values | Description |
|-----------|--------|-------------|
| `layout` | `"nd"` | Data layout |
| `data_select_bit` | `true`/`false` | Data selection |
| `ub_pad` | `true`/`false` | Enable UB padding |

---

### `pto.copy_ubuf_to_ubuf`

- **syntax:**
```mlir
pto.copy_ubuf_to_ubuf %source, %dest, %sid, %n_burst, %len_burst, %src_stride, %dst_stride
    : !llvm.ptr<6>, !llvm.ptr<6>, i64 x5
```
- **CCE:** `__builtin_cce_copy_ubuf_to_ubuf`
- **semantics:** Copy within Unified Buffer.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `%source` | UB source pointer |
| `%dest` | UB destination pointer |
| `%sid` | Stream ID |
| `%n_burst` | Number of bursts |
| `%len_burst` | Length per burst |
| `%src_stride` | Source stride |
| `%dst_stride` | Destination stride |

---

### `pto.copy_ubuf_to_gm`

- **syntax:**
```mlir
pto.copy_ubuf_to_gm %source, %dest, %valid_rows, %valid_cols, %sid, %n_burst, %len_burst,
    %reserved, %burst_dst_stride, %burst_src_stride
    {layout = "LAYOUT"}
    : !llvm.ptr<6>, !llvm.ptr<1>, i64 x8
```
- **CCE:** `__builtin_cce_copy_ubuf_to_gm_align_v2`
- **semantics:** DMA transfer from Unified Buffer (AS=6) to Global Memory (AS=1).

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `%source` | UB source pointer (`!llvm.ptr<6>`) |
| `%dest` | GM destination pointer (`!llvm.ptr<1>`) |
| `%valid_rows` | Number of valid rows |
| `%valid_cols` | Number of valid columns (bytes) |
| `%sid` | Stream ID |
| `%n_burst` | Number of bursts |
| `%len_burst` | Length per burst |
| `%reserved` | Reserved field |
| `%burst_dst_stride` | Destination stride per burst |
| `%burst_src_stride` | Source stride per burst |

---

## Typical DMA Pattern

```mlir
// Configure strides for 2D tile load (4KB rows)
pto.set_loop2_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop1_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64

// Execute GM→UB transfer
pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr, %c32_i64, %c32_i64, %c0_i64, %c1_i64, %c128_i64,
    %c0_i64, %c0_i64, %c0_i64, %c128_i64, %c128_i64
    {layout = "nd", data_select_bit = false, ub_pad = false}
    : !llvm.ptr<1>, !llvm.ptr<6>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64

// Signal MTE2→Vector
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

// ... Vector computation ...

// Configure strides for store
pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop2_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64

// Execute UB→GM transfer
pto.copy_ubuf_to_gm %ub_out, %gm_out, %c32_i64, %c32_i64, %c0_i64, %c1_i64, %c128_i64,
    %c0_i64, %c128_i64, %c128_i64
    {layout = "nd"}
    : !llvm.ptr<6>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64
```
