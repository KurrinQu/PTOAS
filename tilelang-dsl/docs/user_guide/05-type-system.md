## Type System

### Scalar Types

| DSL Type | Description | Bit Width |
|----------|-------------|-----------|
| `pto.i1` | Boolean | 1 |
| `pto.i8` | 8-bit integer | 8 |
| `pto.i16` | 16-bit integer | 16 |
| `pto.i32` | 32-bit integer | 32 |
| `pto.i64` | 64-bit integer | 64 |
| `pto.f16` | Half precision float | 16 |
| `pto.bf16` | Brain float 16 | 16 |
| `pto.f32` | Single precision float | 32 |

Python literals are automatically typed:
- `bool` → `pto.i1`
- `int` → Context-dependent (typically `pto.i32` or `pto.i64`)
- `float` → `pto.f32`

For explicit typing, use type constructors:
```python
x = pto.i32(1024)      # Explicit i32 constant
y: pto.i32 = 1024      # Type annotation
```

### Vector Types

Vector registers have fixed 256-byte width:

```python
v64_f32 = pto.vreg(64, pto.f32)    # 64 lanes of f32 (64 * 32b = 2048b)
v128_f16 = pto.vreg(128, pto.f16)  # 128 lanes of f16 (128 * 16b = 2048b)
```

Constraint: `lanes × bitwidth(element_type) = 2048`

### Typed Masks

Masks are typed by their bit granularity:

| DSL Type | VPTO Type | Description |
|----------|-----------|-------------|
| `pto.mask_b8` | `!pto.mask<b8>` | 8-bit granularity mask |
| `pto.mask_b16` | `!pto.mask<b16>` | 16-bit granularity mask |
| `pto.mask_b32` | `!pto.mask<b32>` | 32-bit granularity mask |

Mask operations must match the vector element family:
- `f32` vectors use `mask_b32`
- `f16` vectors use `mask_b16`
- `i8` vectors use `mask_b8`

```python
# Correct: f32 vector with b32 mask
mask32 = pto.make_mask(pto.f32, PAT.ALL)
vec_f32 = pto.vlds(ptr, offset)
out = pto.vabs(vec_f32, mask32)

# Error: mismatched mask granularity
mask16 = pto.make_mask(pto.f16, PAT.ALL)
out = pto.vabs(vec_f32, mask16)  # Type error!
```

### Pointer Types [Advanced Tier]

Pointers combine element type and memory space:

```python
from pto import MemorySpace

ptr_gm = pto.ptr(pto.f32, MemorySpace.GM)    # GM pointer to f32
ptr_ub = pto.ptr(pto.f16, MemorySpace.UB)    # UB pointer to f16
```

The `MemorySpace` enum provides type-safe memory space specification:

| Enum Value | Description |
|------------|-------------|
| `MemorySpace.GM` | Global Memory (off-chip HBM/DDR) |
| `MemorySpace.UB` | Unified Buffer (on-chip SRAM, 256KB) |

This replaces string literals (`MemorySpace.GM`/`MemorySpace.UB`) with compile-time checked enums.

### Pointer Type Aliases [Advanced Tier]

For clarity in API documentation, the following type alias is used:

| Alias | Equivalent Type | Description |
|-------|----------------|-------------|
| `Tile` | `pto.tile(...)` | Tile buffer with layout and configuration |

### MemRef Types

For buffer-like authoring, use memref types:

```python
buf1d = pto.memref(256, pto.f32, MemorySpace.UB)          # 1D: 256-element f32 buffer in UB
buf2d = pto.memref((256, 128), pto.f32, MemorySpace.UB)   # 2D: 256x128 f32 buffer in UB
```

- **1D shapes**: Use a scalar integer (e.g., `256`)
- **Multi-dimensional shapes**: Use a tuple (e.g., `(256, 128)`)

MemRefs are used for stateless load/store operations that accept `buf_like` operands in VPTO.


