# 4. Predicate Load/Store

> **Category:** UB ↔ Predicate Register data movement
> **Pipeline:** PIPE_V (Vector Core)

Predicate registers (`!pto.mask`) are 256-bit registers that enable per-lane conditional execution. These ops move predicate values between UB and predicate registers.

---

## Predicate Loads

### `pto.vplds`

- **syntax:** `%result = pto.vplds %source[%offset] {dist = "DIST"} : !llvm.ptr<6> -> !pto.mask`
- **CCE:** `__builtin_cce_plds_b8`
- **semantics:** Load predicate register with scalar offset.

**Distribution modes:** `NORM`, `US`, `DS`

**Example:**
```mlir
%mask = pto.vplds %ub[%c0] {dist = "NORM"} : !llvm.ptr<6> -> !pto.mask
```

---

### `pto.vpld`

- **syntax:** `%result = pto.vpld %source[%offset], "DIST" : !llvm.ptr<6>, index -> !pto.mask`
- **CCE:** `__builtin_cce_pld_b8`
- **semantics:** Load predicate register with areg offset.

---

### `pto.vpldi`

- **syntax:** `%result = pto.vpldi %source, %offset, "DIST" : !llvm.ptr<6>, i32 -> !pto.mask`
- **CCE:** `__builtin_cce_pldi_b8`
- **semantics:** Load predicate register with immediate offset.

---

## Predicate Stores

### `pto.vpsts`

- **syntax:** `pto.vpsts %value, %dest[%offset] : !pto.mask, !llvm.ptr<6>`
- **CCE:** `__builtin_cce_psts_b8`
- **semantics:** Store predicate register with scalar offset.

**Example:**
```mlir
pto.vpsts %mask, %ub[%c0] : !pto.mask, !llvm.ptr<6>
```

---

### `pto.vpst`

- **syntax:** `pto.vpst %value, %dest[%offset], "DIST" : !pto.mask, !llvm.ptr<6>, index`
- **CCE:** `__builtin_cce_pst_b8`
- **semantics:** Store predicate register with areg offset.

**Distribution modes:** `NORM`, `PK`

---

### `pto.vpsti`

- **syntax:** `pto.vpsti %value, %dest, %offset, "DIST" : !pto.mask, !llvm.ptr<6>, i32`
- **CCE:** `__builtin_cce_psti_b8`
- **semantics:** Store predicate register with immediate offset.

---

### `pto.vpstu`

- **syntax:** `%align_out, %base_out = pto.vpstu %align_in, %value, %base : !pto.align, !pto.mask, !llvm.ptr<6> -> !pto.align, !llvm.ptr<6>`
- **CCE:** `__builtin_cce_pstu_b16`, `__builtin_cce_pstu_b32`
- **semantics:** Predicate unaligned store with align state update.

---

## Typical Usage Pattern

```mlir
// Generate comparison mask
%mask = pto.vcmp %v0, %v1, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask

// Store mask to UB for later use
pto.vpsts %mask, %ub_mask[%c0] : !pto.mask, !llvm.ptr<6>

// ... later in another kernel ...

// Load mask from UB
%saved_mask = pto.vplds %ub_mask[%c0] {dist = "NORM"} : !llvm.ptr<6> -> !pto.mask

// Use for predicated select
%result = pto.vsel %v_true, %v_false, %saved_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```
