# 4. Predicate Load/Store

> **Category:** UB ↔ Predicate Register data movement
> **Pipeline:** PIPE_V (Vector Core)

Predicate registers (`!pto.mask`) are 256-bit registers that enable per-lane conditional execution. These ops move predicate values between UB and predicate registers.

---

## Predicate Loads

### `pto.vplds`

- **syntax:** `%result = pto.vplds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.mask`
- **semantics:** Load predicate register with scalar offset.

**Distribution modes:** `NORM`, `US`, `DS`

**Example:**
```mlir
%mask = pto.vplds %ub[%c0] {dist = "NORM"} : !pto.ptr<T, ub> -> !pto.mask
```

---

### `pto.vpld`

- **syntax:** `%result = pto.vpld %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.mask`
- **semantics:** Load predicate register with areg offset.

---

### `pto.vpldi`

- **syntax:** `%result = pto.vpldi %source, %offset, "DIST" : !pto.ptr<T, ub>, i32 -> !pto.mask`
- **semantics:** Load predicate register with immediate offset.

---

## Predicate Stores

### `pto.vpsts`

- **syntax:** `pto.vpsts %value, %dest[%offset] : !pto.mask, !pto.ptr<T, ub>`
- **semantics:** Store predicate register with scalar offset.

**Example:**
```mlir
pto.vpsts %mask, %ub[%c0] : !pto.mask, !pto.ptr<T, ub>
```

---

### `pto.vpst`

- **syntax:** `pto.vpst %value, %dest[%offset], "DIST" : !pto.mask, !pto.ptr<T, ub>, index`
- **semantics:** Store predicate register with areg offset.

**Distribution modes:** `NORM`, `PK`

---

### `pto.vpsti`

- **syntax:** `pto.vpsti %value, %dest, %offset, "DIST" : !pto.mask, !pto.ptr<T, ub>, i32`
- **semantics:** Store predicate register with immediate offset.

---

### `pto.vpstu`

- **syntax:** `%align_out, %base_out = pto.vpstu %align_in, %value, %base : !pto.align, !pto.mask, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>`
- **semantics:** Predicate unaligned store with align state update.

---

## Typical Usage Pattern

```mlir
// Generate comparison mask
%mask = pto.vcmp %v0, %v1, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask

// Store mask to UB for later use
pto.vpsts %mask, %ub_mask[%c0] : !pto.mask, !pto.ptr<T, ub>

// ... later in another kernel ...

// Load mask from UB
%saved_mask = pto.vplds %ub_mask[%c0] {dist = "NORM"} : !pto.ptr<T, ub> -> !pto.mask

// Use for predicated select
%result = pto.vsel %v_true, %v_false, %saved_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```
