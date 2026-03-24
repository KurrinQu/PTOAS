# 5. Materialization & Predicate Ops

> **Category:** Scalar broadcast, predicate generation and manipulation
> **Pipeline:** PIPE_V (Vector Core)

These ops create vectors from scalar values and manipulate predicate registers.

---

## Scalar Materialization

### `pto.vbr`

- **syntax:** `%result = pto.vbr %value : T -> !pto.vreg<NxT>`
- **CCE:** broadcast/materialization family
- **semantics:** Broadcast scalar to all vector lanes.

```c
for (int i = 0; i < N; i++)
    dst[i] = value;
```

**Example:**
```mlir
%one = pto.vbr %c1_f32 : f32 -> !pto.vreg<64xf32>
```

---

### `pto.vdup`

- **syntax:** `%result = pto.vdup %input {position = "POSITION", mode = "MODE"} : T|!pto.vreg<NxT> -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vdup_*`
- **semantics:** Duplicate scalar or vector element to all lanes.

```c
for (int i = 0; i < N; i++)
    dst[i] = input_scalar_or_element;
```

---

### `pto.vdupi`

- **syntax:** `%result = pto.vdupi %imm : i32 -> !pto.vreg<NxT>`
- **CCE:** immediate broadcast family
- **semantics:** Broadcast immediate constant to all lanes.

```c
for (int i = 0; i < N; i++)
    dst[i] = (T)imm;
```

---

## Predicate Generation

### `pto.vpset_b8` / `pto.vpset_b16` / `pto.vpset_b32`

- **syntax:** `%result = pto.vpset_b32 "PAT_*" : !pto.mask`
- **CCE:** `__builtin_cce_pset_b8/b16/b32`
- **semantics:** Generate predicate from pattern.

**Patterns:**

| Pattern | Description |
|---------|-------------|
| `PAT_ALL` | All lanes active |
| `PAT_ALLF` | All lanes inactive |
| `PAT_H` | High half active |
| `PAT_Q` | Upper quarter active |
| `PAT_VL1`...`PAT_VL128` | First N lanes active |
| `PAT_M3`, `PAT_M4` | Modular patterns |

**Example — All 64 f32 lanes active:**
```mlir
%all_active = pto.vpset_b32 "PAT_ALL" : !pto.mask
```

**Example — First 16 lanes active:**
```mlir
%first_16 = pto.vpset_b32 "PAT_VL16" : !pto.mask
```

---

### `pto.vpge_b8` / `pto.vpge_b16` / `pto.vpge_b32`

- **syntax:** `%result = pto.vpge_b32 "PAT_*" : !pto.mask`
- **CCE:** `__builtin_cce_pge_b8/b16/b32`
- **semantics:** Generate tail mask — first N lanes active.

```c
for (int i = 0; i < TOTAL_LANES; i++)
    mask[i] = (i < len);
```

**Example — Tail mask for remainder loop:**
```mlir
%tail_mask = pto.vpge_b32 "PAT_VL8" : !pto.mask
```

---

## Predicate Pack/Unpack

### `pto.vppack`

- **syntax:** `%result = pto.vppack %input, "PART" : !pto.mask -> !pto.mask`
- **CCE:** `ppack(...)`
- **semantics:** Narrowing pack of predicate register.

**Part tokens:** `LOWER`, `HIGHER`

---

### `pto.vpunpack`

- **syntax:** `%result = pto.vpunpack %input, "PART" : !pto.mask -> !pto.mask`
- **CCE:** `punpack(...)`
- **semantics:** Widening unpack of predicate register.

---

## Predicate Logical Ops

### `pto.vpand`

- **syntax:** `%result = pto.vpand %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **semantics:** Predicate bitwise AND.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src0[i] & src1[i];
```

---

### `pto.vpor`

- **syntax:** `%result = pto.vpor %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **semantics:** Predicate bitwise OR.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src0[i] | src1[i];
```

---

### `pto.vpxor`

- **syntax:** `%result = pto.vpxor %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **semantics:** Predicate bitwise XOR.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src0[i] ^ src1[i];
```

---

### `pto.vpnot`

- **syntax:** `%result = pto.vpnot %input, %mask : !pto.mask, !pto.mask -> !pto.mask`
- **CCE:** `pnot(...)`
- **semantics:** Predicate bitwise NOT.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = ~src[i];
```

---

### `pto.vpsel`

- **syntax:** `%result = pto.vpsel %src0, %src1, %sel : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **CCE:** `psel(...)`
- **semantics:** Predicate select (mux).

```c
for (int i = 0; i < N; i++)
    dst[i] = sel[i] ? src0[i] : src1[i];
```

---

## Predicate Movement

### `pto.vpmov`

- **syntax:** `%result = pto.vpmov %src, %mask : !pto.mask, !pto.mask -> !pto.mask`
- **semantics:** Predicate move (copy under mask).

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src[i];
```

---

### `pto.vpintlv`

- **syntax:** `%low, %high = pto.vpintlv %src0, %src1 : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- **semantics:** Predicate interleave.

---

### `pto.vpdintlv`

- **syntax:** `%low, %high = pto.vpdintlv %src0, %src1 : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- **semantics:** Predicate deinterleave.

---

### `pto.vpslide`

- **syntax:** `%result = pto.vpslide %src0, %src1, %amt : !pto.mask, !pto.mask, i16 -> !pto.mask`
- **semantics:** Predicate slide/shift.

---

## Typical Usage

```mlir
// Generate all-active mask for f32 (64 lanes)
%all = pto.vpset_b32 "PAT_ALL" : !pto.mask

// Generate tail mask for remainder (last 12 elements)
%tail = pto.vpge_b32 "PAT_VL12" : !pto.mask

// Compare and generate mask
%cmp_mask = pto.vcmp %a, %b, %all, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask

// Combine masks: only process tail elements that passed comparison
%combined = pto.vpand %cmp_mask, %tail, %all : !pto.mask, !pto.mask, !pto.mask -> !pto.mask

// Use for predicated operation
%result = pto.vsel %true_vals, %false_vals, %combined : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```
