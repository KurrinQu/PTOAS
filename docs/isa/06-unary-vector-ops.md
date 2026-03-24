# 6. Unary Vector Ops

> **Category:** Single-input vector operations
> **Pipeline:** PIPE_V (Vector Core)

Element-wise operations that take one vector input and produce one vector output.

---

## Arithmetic

### `pto.vabs`

- **syntax:** `%result = pto.vabs %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vabs_*`
- **A5 types:** i8-i32, f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < 0) ? -src[i] : src[i];
```

---

### `pto.vneg`

- **syntax:** `%result = pto.vneg %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vneg_*`
- **A5 types:** i8-i32, f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = -src[i];
```

---

## Transcendental

### `pto.vexp`

- **syntax:** `%result = pto.vexp %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vexp_*`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i]);
```

---

### `pto.vln`

- **syntax:** `%result = pto.vln %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vln_*`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = logf(src[i]);
```

---

### `pto.vsqrt`

- **syntax:** `%result = pto.vsqrt %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vsqrt_*`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = sqrtf(src[i]);
```

---

### `pto.vrsqrt`

- **syntax:** `%result = pto.vrsqrt %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vrsqrt_*`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = 1.0f / sqrtf(src[i]);
```

---

### `pto.vrec`

- **syntax:** `%result = pto.vrec %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vrec_*`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = 1.0f / src[i];
```

---

## Activation

### `pto.vrelu`

- **syntax:** `%result = pto.vrelu %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vrelu_*`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > 0) ? src[i] : 0;
```

---

## Bitwise

### `pto.vnot`

- **syntax:** `%result = pto.vnot %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vnot_*`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = ~src[i];
```

---

### `pto.vbcnt`

- **syntax:** `%result = pto.vbcnt %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vbcnt_*`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = __builtin_popcount(src[i]);
```

---

### `pto.vcls`

- **syntax:** `%result = pto.vcls %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **CCE:** `__builtin_cce_vcls_*`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = count_leading_sign_bits(src[i]);
```

---

## Movement

### `pto.vmov`

- **syntax:** `%result = pto.vmov %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **semantics:** Vector register copy.

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i];
```

---

## Typical Usage

```mlir
// Softmax numerator: exp(x - max)
%sub = pto.vsub %x, %max_broadcast : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>
%exp = pto.vexp %sub : !pto.vreg<64xf32> -> !pto.vreg<64xf32>

// Reciprocal for division
%sum_rcp = pto.vrec %sum : !pto.vreg<64xf32> -> !pto.vreg<64xf32>

// ReLU activation
%activated = pto.vrelu %linear_out : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
```
