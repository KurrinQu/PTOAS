# 3. Vector Load/Store

> **Category:** UB ↔ Vector Register data movement
> **Pipeline:** PIPE_V (Vector Core)

Vector loads move data from Unified Buffer (UB) to vector registers (`vreg`). Vector stores move data from `vreg` back to UB. All vector compute operates only on `vreg` — UB is the staging area between DMA and compute.

---

## Latency and throughput (A5)

**Cycle-accurate simulator (CA model)** issue→retire timings for vector-side instructions behind this chapter. Values are **simulator** results, **not** guaranteed for silicon.

**SOC:** Tables below are from **Ascend910_9599** CA sim (the pto-isa ST default when **Ascend950PR_9599** is not selected).

**Log `dist:` tokens:** PTO load/store modes lower to **`RV_VLD` / `RV_VLDI` / `RV_VST` / `RV_VSTI`** with a **`dist:`** field on the vector pipes (`RVECLD` / `RVECST`). Some simulator logs typo contiguous load as `dist:NORAML`; treat as **`NORMAL`**.

### Reference op latencies (A5 mnemonics)

| A5 mnemonic | Mode / note | Typical issue→retire (cycles) |
|-------------|-------------|------------------------------|
| `RV_VLD` | `dist:NORMAL` / `NORAML` | **9** |
| `RV_VLDI` | `dist:DINTLV_B32` (dual vreg) | **9** |
| `RV_VST` / `RV_VSTI` | `dist:NORM_B32` | **9** |
| `RV_VGATHER2` | `Dtype: B32` | **27–28** |
| `RV_VGATHERB` | indexed byte gather | **~21** |
| `RV_VSCATTER` | `Dtype: B16` | **~17** |
| `RV_VADD` | F32 between UB-backed ops | **7** |

### `dist:` tokens (issue→retire)

Most **`dist:`** tokens are **9** issue→retire cycles. **`INTLV_*`** on **`RV_VSTI`** are **12** cycles.

| `dist:` (as in log) | RV op | issue→retire (cycles) |
|---------------------|-------|----------------------|
| `DINTLV_B32` | `RV_VLDI` | **9** |
| `DINTLV_B16` | `RV_VLDI` | **9** |
| `DINTLV_B8` | `RV_VLDI` | **9** |
| `BRC_B32` | `RV_VLD` | **9** |
| `BRC_B8` | `RV_VLD` | **9** |
| `BRC_B16` | `RV_VLD` | **9** |
| `BRC_BLK` | `RV_VLD` | **9** |
| `INTLV_B32` | `RV_VSTI` | **12** |
| `INTLV_B16` | `RV_VSTI` | **12** |
| `INTLV_B8` | `RV_VSTI` | **12** |
| `UNPK_B8` | `RV_VLD` | **9** |
| `UNPK_B16` | `RV_VLD` | **9** |
| `UNPK_B32` | `RV_VLD` | **9** |
| `NORM_B32` | `RV_VSTI` | **9** |
| `NORM_B16` | `RV_VSTI` | **9** |
| `NORM_B8` | `RV_VSTI` | **9** |
| `PK_B32` | `RV_VSTI` | **9** |
| `PK_B16` | `RV_VSTI` | **9** |
| `NORMAL` / `NORAML` | `RV_VLD` | **9** |

**Note:** PTO intrinsic **`BLK`** matches the **`BRC_BLK`** `dist:` string on **`RV_VLD`** in simulator logs (block-replicate path; not a plain contiguous copy in the usual tiling use).

**Issue (vector load/store):** `pto.vlds` (**`RV_VLD`**) is **dual-issue capable**: two independent `pto.vlds` can issue **in the same cycle**. **Alternatively**, the hardware can issue **one** `pto.vlds` **and** **one** `pto.vsts` together (**1+1**) in the same cycle. Each cycle is **either** dual **`vlds`** **or** **`vlds` + `vsts` (1+1)**—those two issue modes are mutually exclusive. Sustained throughput still depends on RAW hazards and loop structure.

**Throughput (simulator, pattern-dependent):**

- **`RV_VLD` / `pto.vlds`:** Dual-issue **or** half of a **1+1** with `vsts`, per the rule above.
- **`RV_VST` / `pto.vsts`:** In a **1+1** cycle, pairs with one `vlds`; otherwise typically **one** store per cycle in tight loops.
- **`RV_VGATHER2`:** Much lower than contiguous `RV_VLD` (on the order of **~0.1** ops/cycle in steady-state alongside 27–28-cycle latency).

### PTO `dist` summary (loads)

| PTO `dist` (load) | Latency |
|-------------------|-------------------|
| `NORM` | **9** cycles |
| `UNPK_B8`, `UNPK_B16`, `UNPK_B32` | **9** cycles |
| `DINTLV_B32` | **9** cycles (`RV_VLDI`) |
| `DINTLV_B16`, `DINTLV_B8` | **9** cycles (same `RV_VLDI` + `dist:DINTLV_*` path as `DINTLV_B32`) |
| `BRC_B32` | **9** cycles |
| `BRC_B8`, `BRC_B16` | **9** cycles (`RV_VLD`) |
| `BLK` | **9** cycles as **`dist:BRC_BLK`** on `RV_VLD` |
| `BDINTLV` | **9** cycles |
| `US_*`, `DS_*`, `SPLT*` | **9** cycles |

### PTO `dist` summary (stores)

| PTO `dist` (store) | Latency |
|--------------------|-------------------|
| `NORM_B8`, `NORM_B16`, `NORM_B32` | **9** cycles (`RV_VSTI`) |
| `PK_B16`, `PK_B32` | **9** cycles |
| `INTLV_B32` (`pto.vstx2`) | **12** cycles |
| `INTLV_B16`, `INTLV_B8` | **12** cycles (same interleave store path as `INTLV_B32`) |
| `MRG4CHN_B8`, `MRG2CHN_*` | **9** cycles |

### Gather, scatter, and special addressing

| PTO op | A5-level | Latency |
|--------|----------|-------------------|
| `pto.vgather2` | `RV_VGATHER2` | **27–28** cycles (pattern-dependent) |
| `pto.vgatherb` | `RV_VGATHERB` | **~21** cycles issue→retire |
| `pto.vgather2_bc` | (broadcast gather) | **27–28** cycles (same as **`pto.vgather2`**) |
| `pto.vscatter` | `RV_VSCATTER` | **~17** cycles for **`Dtype: B16`** |

### Strided loads/stores, unaligned ops, alignment state

Ops such as **`pto.vldas`**, **`pto.vldus`**, **`pto.vsld`**, **`pto.vsldb`**, **`pto.vsst`**, **`pto.vsstb`**, **`pto.vsta`**, **`pto.vstas`**, **`pto.vstar`**, **`pto.vstu`**, **`pto.vstus`**, **`pto.vstur`**: **9** cycles (same vector load/store pipe family as contiguous `RV_VLD` / `RV_VST` unless listed otherwise above).

### Dual-issue vs DMA

DMA **`TLOAD` / `TSTORE`** (global memory ↔ UB) use **MTE** pipes, not `RV_VLD`/`RV_VST`. **MTE2** `MOV_*` latency is not the same as vector `RV_VLD` latency (see `02-dma-copy.md` for GM↔UB movement).

---

## Contiguous Loads

### `pto.vlds`

- **syntax:** `%result = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>`
- **semantics:** Vector load with distribution mode.

**Distribution modes:**

| Mode | Description | C Semantics | Latency |
|------|-------------|-------------|---------------------|
| `NORM` | Contiguous 256B load | `dst[i] = UB[base + i * sizeof(T)]` | **9** cycles |
| `BRC_B32` | Broadcast single element | `dst[i] = UB[base]` for all i | **9** cycles |
| `BRC_B8`, `BRC_B16` | Broadcast first lane element | Same idea at B8/B16 width | **9** cycles |
| `US_B8/B16` | Upsample (duplicate each element) | `dst[2*i] = dst[2*i+1] = UB[base + i]` | **9** cycles |
| `DS_B8/B16` | Downsample (every 2nd element) | `dst[i] = UB[base + 2*i]` | **9** cycles |
| `UNPK_B8/B16/B32` | Unpack (zero-extend to wider type) | `dst_i32[i] = (uint32_t)UB_i16[base + 2*i]` | **9** cycles |
| `SPLT4CHN_B8` | Split 4-channel (RGBA → R plane) | Extract every 4th byte | **9** cycles |
| `SPLT2CHN_B8/B16` | Split 2-channel | Extract every 2nd element | **9** cycles |
| `DINTLV_B32` | Deinterleave 32-bit | Even elements only | **9** cycles |
| `DINTLV_B16`, `DINTLV_B8` | Deinterleave 16-bit / 8-bit | Pair lanes from interleaved UB | **9** cycles |
| `BDINTLV` | Block deinterleave | (see PTO headers for exact tiling) | **9** cycles |
| `BLK` | Block load | Blocked / tiled access pattern (see PTO headers) | **9** cycles (`dist:BRC_BLK` on `RV_VLD`) |

**Example — Contiguous load:**
```mlir
%v = pto.vlds %ub[%offset] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

**Example — Broadcast scalar to all lanes:**
```mlir
%v = pto.vlds %ub[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

---

### `pto.vldas`

- **syntax:** `%result = pto.vldas %source : !pto.ptr<T, ub> -> !pto.align`
- **semantics:** Prime alignment buffer for subsequent unaligned load.
- **Latency:** **9** cycles.

---

### `pto.vldus`

- **syntax:** `%result, %align_out, %base_out = pto.vldus %source, %align : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align, !pto.ptr<T, ub>`
- **semantics:** Unaligned load using primed align state.
- **Latency:** **9** cycles.

**Unaligned load pattern:**
```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align2, %ub2 = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align, !pto.ptr<f32, ub>
```

---

## Dual Loads (Deinterleave)

### `pto.vldx2`

- **syntax:** `%low, %high = pto.vldx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Dual load with deinterleave (AoS → SoA conversion).
- **Latency:** **`DINTLV_B32` → 9** cycles on `RV_VLDI`. **`DINTLV_B16` / `DINTLV_B8` → 9** cycles on `RV_VLDI`. **`BDINTLV` → 9** cycles on `RV_VLDI`.

**Distribution modes:** `DINTLV_B8`, `DINTLV_B16`, `DINTLV_B32`, `BDINTLV`

```c
// DINTLV_B32: deinterleave 32-bit elements
for (int i = 0; i < 64; i++) {
    low[i]  = UB[base + 8*i];       // even elements
    high[i] = UB[base + 8*i + 4];   // odd elements
}
```

**Example — Load interleaved XY pairs into separate X/Y vectors:**
```mlir
%x, %y = pto.vldx2 %ub[%offset], "DINTLV_B32" : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
```

---

## Strided Loads

### `pto.vsld`

- **syntax:** `%result = pto.vsld %source[%offset], "STRIDE" : !pto.ptr<T, ub> -> !pto.vreg<NxT>`
- **semantics:** Strided load with fixed stride pattern.
- **Latency:** **9** cycles.

**Stride modes:** `STRIDE_S3_B16`, `STRIDE_S4_B64`, `STRIDE_S8_B32`, `STRIDE_S2_B64`

---

### `pto.vsldb`

- **syntax:** `%result = pto.vsldb %source, %offset, %mask : !pto.ptr<T, ub>, i32, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Block-strided load for 2D tile access.
- **Latency:** **9** cycles.

---

## Gather (Indexed) Loads

### `pto.vgather2`

- **syntax:** `%result = pto.vgather2 %source, %offsets, %active_lanes : !pto.ptr<T, ub>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- **semantics:** Indexed gather from UB.
- **Latency:** **27–28** cycles per `RV_VGATHER2`; throughput much lower than contiguous `RV_VLD` (see **Latency and throughput (A5)** at the start of this chapter).

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i] * sizeof(T)];
```

---

### `pto.vgatherb`

- **syntax:** `%result = pto.vgatherb %source, %offsets, %active_lanes : !pto.ptr<T, ub>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- **semantics:** Byte-granularity indexed gather from UB.
- **Latency:** **~21** cycles issue→retire.

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i]];  // byte-addressed
```

---

### `pto.vgather2_bc`

- **syntax:** `%result = pto.vgather2_bc %source, %offsets, %mask : !pto.ptr<T, ub>, !pto.vreg<NxI>, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Gather with broadcast, conditioned by mask.
- **Latency:** **27–28** cycles (same as **`pto.vgather2`**).

---

## Contiguous Stores

### `pto.vsts`

- **syntax:** `pto.vsts %value, %dest[%offset], %mask {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.mask`
- **semantics:** Vector store with distribution mode.

**Distribution modes:**

| Mode | Description | C Semantics | Latency |
|------|-------------|-------------|---------------------|
| `NORM_B8/B16/B32` | Contiguous store | `UB[base + i] = src[i]` | **9** cycles |
| `PK_B16/B32` | Pack/narrowing store | `UB_i16[base + 2*i] = truncate_16(src_i32[i])` | **9** cycles |
| `MRG4CHN_B8` | Merge 4 channels (R,G,B,A → RGBA) | Interleave 4 planes | **9** cycles |
| `MRG2CHN_B8/B16` | Merge 2 channels | Interleave 2 planes | **9** cycles |

**Example — Contiguous store:**
```mlir
pto.vsts %v, %ub[%offset], %mask {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
```

---

## Dual Stores (Interleave)

### `pto.vstx2`

- **syntax:** `pto.vstx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask`
- **semantics:** Dual interleaved store (SoA → AoS conversion).
- **Latency:** **`INTLV_B32` / `INTLV_B16` / `INTLV_B8` → 12** cycles on `RV_VSTI`.

**Distribution modes:** `INTLV_B8`, `INTLV_B16`, `INTLV_B32`

```c
// INTLV_B32:
for (int i = 0; i < 64; i++) {
    UB[base + 8*i]     = low[i];
    UB[base + 8*i + 4] = high[i];
}
```

---

## Strided Stores

### `pto.vsst`

- **syntax:** `pto.vsst %value, %dest[%offset], "STRIDE" : !pto.vreg<NxT>, !pto.ptr<T, ub>`
- **semantics:** Strided store with fixed stride pattern.
- **Latency:** **9** cycles.

---

### `pto.vsstb`

- **syntax:** `pto.vsstb %value, %dest, %offset, %mask : !pto.vreg<NxT>, !pto.ptr<T, ub>, i32, !pto.mask`
- **semantics:** Block-strided store for 2D tile access.
- **Latency:** **9** cycles.

---

## Scatter (Indexed) Stores

### `pto.vscatter`

- **syntax:** `pto.vscatter %value, %dest, %offsets, %active_lanes : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.vreg<NxI>, index`
- **semantics:** Indexed scatter to UB.
- **Latency:** **~17** cycles for **`Dtype: B16`**.

```c
for (int i = 0; i < active_lanes; i++)
    UB[base + offsets[i] * sizeof(T)] = src[i];
```

---

## Alignment State Stores

### `pto.vsta`

- **syntax:** `pto.vsta %value, %dest[%offset] : !pto.align, !pto.ptr<T, ub>, index`
- **semantics:** Flush alignment state to memory.
- **Latency:** **9** cycles.

---

### `pto.vstas`

- **syntax:** `pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32`
- **semantics:** Flush alignment state with scalar offset.
- **Latency:** **9** cycles.

---

### `pto.vstar`

- **syntax:** `pto.vstar %value, %dest : !pto.align, !pto.ptr<T, ub>`
- **semantics:** Flush remaining alignment state.
- **Latency:** **9** cycles.

---

## Stateful Store Ops

These ops make reference-updated state explicit as SSA results.

### `pto.vstu`

- **syntax:** `%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base, "MODE" : !pto.align, index, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, index`
- **semantics:** Unaligned store with align + offset state update.
- **Latency:** **9** cycles.

**Mode tokens:** `POST_UPDATE`, `NO_POST_UPDATE`

---

### `pto.vstus`

- **syntax:** `%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base, "MODE" : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>`
- **semantics:** Unaligned store with scalar offset and state update.
- **Latency:** **9** cycles.

---

### `pto.vstur`

- **syntax:** `%align_out = pto.vstur %align_in, %value, %base, "MODE" : !pto.align, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align`
- **semantics:** Unaligned store with residual flush and state update.
- **Latency:** **9** cycles.
