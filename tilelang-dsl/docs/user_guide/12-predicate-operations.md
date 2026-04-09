### Predicate Operations

Operations for creating and manipulating typed masks.

**Recommended API**: For most use cases, prefer the unified `pto.make_mask()` function which automatically selects the appropriate mask granularity based on element type and supports both tail processing (remaining element count) and pattern-based mask generation. This eliminates the need to manually choose between `plt_b8`/`plt_b16`/`plt_b32` (tail processing) and `pset_b8`/`pset_b16`/`pset_b32` (pattern generation) operations.

**Pattern alias**: For brevity in examples, the documentation uses `PAT` as an alias for `pto.MaskPattern` (e.g., `PAT.ALL` instead of `pto.MaskPattern.PAT_ALL`). In practice, you can create this alias with `from pto import MaskPattern as PAT` or `PAT = pto.MaskPattern`.

#### `pto.pset_b8(pattern: pto.MaskPattern) -> pto.mask_b8`

**Description**: Creates an 8-bit granularity mask from a pattern.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `pto.MaskPattern` | Mask pattern enum (e.g., `pto.MaskPattern.PAT_ALL`, `pto.MaskPattern.PAT_EVEN`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b8` | 8-bit granularity mask |

**Constraints**:
- Used with `i8` vector operations

**Example**:
```python
mask8 = pto.make_mask(pto.i8, PAT.ALL)
```

#### `pto.pset_b16(pattern: pto.MaskPattern) -> pto.mask_b16`

**Description**: Creates a 16-bit granularity mask from a pattern.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `pto.MaskPattern` | Mask pattern enum (e.g., `pto.MaskPattern.PAT_ALL`, `pto.MaskPattern.PAT_EVEN`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b16` | 16-bit granularity mask |

**Constraints**:
- Used with `f16`/`bf16`/`i16` vector operations

**Example**:
```python
mask16 = pto.make_mask(pto.f16, PAT.ALL)
```

#### `pto.pset_b32(pattern: pto.MaskPattern) -> pto.mask_b32`

**Description**: Creates a 32-bit granularity mask from a pattern.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `pto.MaskPattern` | Mask pattern enum (e.g., `pto.MaskPattern.PAT_ALL`, `pto.MaskPattern.PAT_EVEN`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b32` | 32-bit granularity mask |

**Constraints**:
- Used with `f32`/`i32` vector operations

**Example**:
```python
mask32 = pto.make_mask(pto.f32, PAT.ALL)
```

#### `pto.pge_b8(vec: VRegType, scalar: ScalarType) -> pto.mask_b8`

**Description**: Creates 8-bit mask where vector elements ≥ scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (element type must match mask granularity) |
| `scalar` | `ScalarType` | Scalar comparison value |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b8` | 8-bit granularity mask |

**Constraints**:
- Vector element type must be `i8` or compatible

#### `pto.pge_b16(vec: VRegType, scalar: ScalarType) -> pto.mask_b16`

**Description**: Creates 16-bit mask where vector elements ≥ scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (element type must match mask granularity) |
| `scalar` | `ScalarType` | Scalar comparison value |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b16` | 16-bit granularity mask |

**Constraints**:
- Vector element type must be `f16`/`bf16`/`i16`

#### `pto.pge_b32(vec: VRegType, scalar: ScalarType) -> pto.mask_b32`

**Description**: Creates 32-bit mask where vector elements ≥ scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (element type must match mask granularity) |
| `scalar` | `ScalarType` | Scalar comparison value |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b32` | 32-bit granularity mask |

**Constraints**:
- Vector element type must be `f32`/`i32`

**Example**:
```python
mask = pto.pge_b32(vec_f32, pto.f32(0.0))
```

#### `pto.plt_b8(vec: VRegType, scalar: ScalarType) -> pto.mask_b8`

**Description**: Creates 8-bit mask where vector elements < scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (element type must match mask granularity) |
| `scalar` | `ScalarType` | Scalar comparison value |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b8` | 8-bit granularity mask |

#### `pto.plt_b16(vec: VRegType, scalar: ScalarType) -> pto.mask_b16`

**Description**: Creates 16-bit mask where vector elements < scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (element type must match mask granularity) |
| `scalar` | `ScalarType` | Scalar comparison value |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b16` | 16-bit granularity mask |

#### `pto.plt_b32(vec: VRegType, scalar: ScalarType) -> (pto.mask_b32, pto.i32)`

**Description**: Creates 32-bit mask where vector elements < scalar, returns mask and remaining count.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (element type must match mask granularity) |
| `scalar` | `ScalarType` | Scalar comparison value |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b32` | 32-bit granularity mask |
| `remaining` | `pto.i32` | Remaining element count |

**Example**:
```python
mask, remaining = pto.plt_b32(vec_f32, pto.f32(10.0))
```

#### `pto.make_mask(element_type: Type, value: pto.i32 | pto.MaskPattern) -> MaskType | (MaskType, pto.i32)`

**Description**: Creates a mask with appropriate bitwidth (8, 16, or 32) based on element type, automatically inferring whether to perform tail processing or pattern-based mask generation based on the `value` parameter type. This convenience function eliminates the need to manually choose between `plt_b8`/`plt_b16`/`plt_b32` and `pset_b8`/`pset_b16`/`pset_b32` operations.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `element_type` | `Type` | Element type (e.g., `pto.f32`, `pto.f16`, `pto.i8`) |
| `value` | `pto.i32` \| `pto.MaskPattern` | Either: <br>- Remaining element count (as `pto.i32`) for tail processing <br>- Mask pattern enum value for fixed mask generation (e.g., `pto.MaskPattern.PAT_ALL`, `pto.MaskPattern.PAT_VL32`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `MaskType` | Generated mask with appropriate granularity |
| `remaining` | `pto.i32` | Updated remaining element count (only returned when `value` is a `pto.i32` for tail processing) |

**Constraints**:
- The `element_type` must be one of: `f32`, `i32`, `f16`, `bf16`, `i16`, `i8`
- The returned mask granularity matches the element type: 32-bit for `f32`/`i32`, 16-bit for `f16`/`bf16`/`i16`, 8-bit for `i8`
- The function infers the operation mode from the `value` parameter type at compile time:
  - `pto.i32` value → tail processing mode (returns `(mask, updated_remaining)`)
  - `pto.MaskPattern` enum value → pattern mode (returns `mask` only)

**Implementation Note**: This function is a DSL macro that performs type-based dispatch at compile time:
- When `value` is a `pto.i32` expression: expands to corresponding `plt_b` instruction (`plt_b32`, `plt_b16`, or `plt_b8`)
- When `value` is a `pto.MaskPattern` enum value: expands to corresponding `pset_b` instruction (`pset_b32`, `pset_b16`, or `pset_b8`)

**Example**:
```python
# Tail processing with f32 vectors: value is pto.i32 → expands to plt_b32
mask_f32, remaining_f32 = pto.make_mask(pto.f32, remaining_elements)

# Tail processing with f16 vectors: value is pto.i32 → expands to plt_b16  
mask_f16, remaining_f16 = pto.make_mask(pto.f16, remaining_elements)

# Tail processing with i8 vectors: value is pto.i32 → expands to plt_b8
mask_i8, remaining_i8 = pto.make_mask(pto.i8, remaining_elements)

# Pattern-based mask with f32 vectors: value is MaskPattern enum → expands to pset_b32
mask_all_f32 = pto.make_mask(pto.f32, PAT.ALL)

# Pattern-based mask with f16 vectors: value is MaskPattern enum → expands to pset_b16  
mask_even_f16 = pto.make_mask(pto.f16, PAT.EVEN)

# Pattern-based mask with i8 vectors: value is MaskPattern enum → expands to pset_b8
mask_all_i8 = pto.make_mask(pto.i8, PAT.ALL)

# Type annotations help clarify expected parameter types
remaining: pto.i32 = 1024
mask1, updated = pto.make_mask(pto.f32, remaining)     # tail processing
mask2 = pto.make_mask(pto.f32, PAT.ALL)              # pattern mode
```

#### `pto.ppack(mask: MaskType, part: str) -> MaskType`

**Description**: Rearranges a mask according to the requested `part` selector.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Input mask (`mask_b8`, `mask_b16`, or `mask_b32`) |
| `part` | `str` | Part selector such as `"PART_EVEN"` or `"PART_ODD"` |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `packed` | `MaskType` | Reordered mask |

#### `pto.punpack(mask: MaskType, part: str) -> MaskType`

**Description**: Applies the inverse mask-part rearrangement selected by `part`.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Input mask |
| `part` | `str` | Part selector such as `"PART_EVEN"` or `"PART_ODD"` |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `MaskType` | Reordered mask |

#### `pto.pnot(mask: MaskType, gate: MaskType) -> MaskType`

**Description**: Predicate negation under a same-granularity mask gate.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Input mask |
| `gate` | `MaskType` | Gating mask with the same granularity |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `negated` | `MaskType` | Negated mask |

#### `pto.psel(src0: MaskType, src1: MaskType, mask: MaskType) -> MaskType`

**Description**: Selects between two masks using a third mask as selector.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `MaskType` | First input mask |
| `src1` | `MaskType` | Second input mask |
| `mask` | `MaskType` | Selection mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `MaskType` | Selected mask |

