# 8. Vec-Scalar Ops

> **Category:** Vector-scalar operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that combine a vector with a scalar value, applying the scalar to every lane.

---

## Arithmetic

### `pto.vadds`

- **syntax:** `%result = pto.vadds %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vadds_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] + scalar;
```

---

### `pto.vsubs`

- **syntax:** `%result = pto.vsubs %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vsubs_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] - scalar;
```

---

### `pto.vmuls`

- **syntax:** `%result = pto.vmuls %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vmuls_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] * scalar;
```

---

### `pto.vmaxs`

- **syntax:** `%result = pto.vmaxs %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vmaxs_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > scalar) ? src[i] : scalar;
```

---

### `pto.vmins`

- **syntax:** `%result = pto.vmins %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vmins_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < scalar) ? src[i] : scalar;
```

---

## Bitwise

### `pto.vands`

- **syntax:** `%result = pto.vands %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vands_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] & scalar;
```

---

### `pto.vors`

- **syntax:** `%result = pto.vors %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vors_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] | scalar;
```

---

### `pto.vxors`

- **syntax:** `%result = pto.vxors %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vxors_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] ^ scalar;
```

---

## Shift

### `pto.vshls`

- **syntax:** `%result = pto.vshls %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vshls_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] << scalar;
```

---

### `pto.vshrs`

- **syntax:** `%result = pto.vshrs %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vshrs_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] >> scalar;
```

---

## Carry Operations

### `pto.vaddcs`

- **syntax:** `%result, %carry = pto.vaddcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **CCE:** `__builtin_cce_vaddcs_*`
- **semantics:** Add with carry-in and carry-out.

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i] + carry_in[i];
    dst[i] = (T)r;
    carry_out[i] = (r >> bitwidth);
}
```

---

### `pto.vsubcs`

- **syntax:** `%result, %borrow = pto.vsubcs %lhs, %rhs, %borrow_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **CCE:** `__builtin_cce_vsubcs_*`
- **semantics:** Subtract with borrow-in and borrow-out.

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i] - borrow_in[i];
    borrow_out[i] = (src0[i] < src1[i] + borrow_in[i]);
}
```

---

## Typical Usage

```mlir
// Add bias to all elements
%biased = pto.vadds %activation, %bias_scalar : !pto.vreg<64xf32>, f32 -> !pto.vreg<64xf32>

// Scale by constant
%scaled = pto.vmuls %input, %scale : !pto.vreg<64xf32>, f32 -> !pto.vreg<64xf32>

// Clamp to [0, 255] for uint8 quantization
%clamped_low = pto.vmaxs %input, %c0 : !pto.vreg<64xf32>, f32 -> !pto.vreg<64xf32>
%clamped = pto.vmins %clamped_low, %c255 : !pto.vreg<64xf32>, f32 -> !pto.vreg<64xf32>

// Shift right by fixed amount
%shifted = pto.vshrs %data, %c4 : !pto.vreg<64xi32>, i32 -> !pto.vreg<64xi32>
```
