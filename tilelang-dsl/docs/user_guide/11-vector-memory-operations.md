### Address Generation Syntax Sugar

To simplify address calculation and reduce manual byte offset computation errors, TileLang DSL provides syntactic sugar for vector load/store operations using element-based indexing. This syntax automatically computes the byte offset based on tile shape, element type, and layout.

#### Indexing Syntax

The syntax supports two indexing modes for different operations:

1. **Vector-range indexing** (for vector load/store operations):
   - **Row-major layout (default)**: `tile[row_index, col_start:]`
     - `row_index`: Row index (0-based)
     - `col_start:`: Starting column index followed by colon, indicating a vector-width contiguous region starting from this column
     - The colon (`:`) indicates an implicit vector-width range determined by hardware vector size (256 bytes) and element type
   
   - **Column-major layout**: `tile[row_start:, col_index]`
     - `row_start:`: Starting row index followed by colon, indicating a vector-width contiguous region starting from this row
     - `col_index`: Column index (0-based)
     - Used for column-major tiles (`BLayout.COL_MAJOR`) where elements are stored column-wise
   
   - **1D tile indexing**: `tile[start:]` (or equivalently `tile[0, start:]` for row-major or `tile[start:, 0]` for column-major)
     - `start:`: Starting element index followed by colon

2. **Single-element indexing** (for scalar load operations like `pto.vsld`):
   - **Row-major layout (default)**: `tile[row_index, col_index]`
     - `row_index`: Row index (0-based)
     - `col_index`: Column index (0-based)
     - Loads a single element at the specified position and broadcasts it to all vector lanes
   
   - **Column-major layout**: `tile[row_index, col_index]` (same syntax)
     - `row_index`: Row index (0-based)
     - `col_index`: Column index (0-based)
     - Same syntax as row-major; the layout determines how the offset is computed
   
   - **1D tile indexing**: `tile[pos]`
     - `pos`: Element index (0-based)
     - Loads a single element at the specified position and broadcasts it to all vector lanes

#### Vector Width Calculation

The number of elements loaded/stored in a single vector operation is determined by:

```
vector_lanes = 256 // element_size_bytes(element_type)
```

**Convenience API**: Use `pto.get_lanes(dtype)` to compute vector lanes for a given element type (e.g., `pto.get_lanes(pto.f32)` returns 64, `pto.get_lanes(pto.f16)` returns 128). See [Type Query Operations](09-frontend-operations.md#type-query-operations) for full documentation.

Where `element_size_bytes` is:
- 1 byte for `i8`
- 2 bytes for `i16`, `f16`, `bf16`
- 4 bytes for `i32`, `f32`
- 8 bytes for `i64`

#### Offset Computation

The byte offset is automatically computed based on tile layout:

- **Row-major layout** (`BLayout.ROW_MAJOR`):
  ```
  offset = (row_index * stride_row + col_start) * element_size_bytes
  ```
  where `stride_row` is the row stride in elements (typically `tile.shape[1]` for contiguous tiles).

- **Column-major layout** (`BLayout.COL_MAJOR`):
  - For syntax `tile[row_start:, col_index]`:
    ```
    offset = (col_index * stride_col + row_start) * element_size_bytes
    ```
  - For backward compatibility with traditional offset calculation:
    ```
    offset = (col_start * stride_col + row_index) * element_size_bytes
    ```
  where `stride_col` is the column stride in elements (typically `tile.shape[0]` for contiguous tiles), `row_start` is the starting row index, and `col_index` is the column index.

**Note**: 
- For single-element indexing (`tile[row, col]` or `tile[pos]`), the same offset formulas apply with `col_start` replaced by `col_index` (or `start` replaced by `pos` for 1D tiles).
- For column-major vector-range indexing (`tile[row_start:, col_index]`), the offset formula uses `row_start` as the starting position along the contiguous dimension.
- The compiler automatically handles the appropriate substitution based on the indexing syntax and tile layout.

#### Constraints

1. **Boundary checks**: The requested region must be within tile bounds:
   - **For vector-range indexing** (`:` syntax):
     - **Row-major layout** (`tile[row_index, col_start:]`):
       - `row_index < tile.shape[0]` and `col_start + vector_lanes <= tile.shape[1]`
     - **Column-major layout** (`tile[row_start:, col_index]`):
       - `row_start + vector_lanes <= tile.shape[0]` and `col_index < tile.shape[1]`
     - **1D tile indexing**: `tile[start:]`
       - `start + vector_lanes <= tile.shape[0]` (or `tile.shape[1]` for 1D tiles)
   - **For single-element indexing** (no `:` syntax):
     - 2D: `row_index < tile.shape[0]` and `col_index < tile.shape[1]` (same for both layouts)
     - 1D: `pos < tile.shape[0]` (or `tile.shape[1]` for 1D tiles)

2. **Alignment**: The computed offset must satisfy hardware alignment requirements for the operation.

3. **Full vectors only**: The `:` syntax always loads/stores a full vector width. For partial vectors, use the traditional byte offset approach with explicit mask handling.

4. **Single-element operations**: The single-element indexing syntax (`tile[row, col]` or `tile[pos]`) is only supported for scalar load operations like `pto.vsld`. For other operations, use vector-range indexing with `:` syntax.

#### Supported Operations

The indexing syntax is supported for all vector load and store operations with the following syntax mapping:

- **Vector-range indexing** (`tile[row, col:]` or `tile[start:]`):
  - Load operations: `vlds`, `vldas`, `vldus`, `vplds`, `vldx2`
  - Store operations: `vsts`, `vsta`, `psts`, `vsst`, `vstx2`

- **Single-element indexing** (`tile[row, col]` or `tile[pos]`):
  - Load operations: `vsld` (scalar load with broadcast)

#### Examples

The following examples use row-major layout syntax. For column-major tiles, use `tile[row_start:, col_index]` syntax instead of `tile[row_index, col_start:]`.

```python
# 2D tile indexing (row-major layout)
vec = pto.vlds(tile[i, j:])          # Load vector from row i, columns j to j+vector_lanes-1
pto.vsts(vec, tile[i, j:], mask)     # Store vector with mask

# 1D tile indexing  
vec = pto.vlds(tile[k:])             # Load vector from elements k to k+vector_lanes-1
pto.vsts(vec, tile[k:], mask)        # Store vector with mask

# Dual load with indexing
vec1, vec2 = pto.vldx2(tile_a[i, j:], tile_b[i, j:])

# Aligned load with indexing
vec = pto.vldas(tile[i, j:], align)

# Scalar load (broadcast)
vec = pto.vsld(tile[i, j])          # Load scalar at tile[i,j] and broadcast to vector
```

#### Comparison with Manual Offset Calculation

**Traditional approach (error-prone):**
```python
# Manual byte offset calculation for f32 tile
rows, cols = tile.shape
row_offset = i * cols * 4  # Hard-coded 4 bytes for f32
col_offset = j * 4
offset = row_offset + col_offset
vec = pto.vlds(tile, offset)
```

**New syntax (type-safe):**
```python
# Automatic offset calculation
vec = pto.vlds(tile[i, j:])  # Compiler computes correct offset for any element type
```

The syntax sugar eliminates manual byte calculations, reduces errors, and makes code generic across different element types (e.g., the same kernel works for both `f16` and `f32` without modification).

### Vector Load Operations

Operations for loading data from memory into vector registers.

#### `pto.vlds(buf: ptr, offset: Index) -> VRegType`  [Advanced Tier]
#### `pto.vlds(tile[row, col:]) -> VRegType`  [Basic Tier]
#### `pto.vlds(tile[start:]) -> VRegType`  [Basic Tier]

**Description**: Stateless vector load from buffer. Supports both traditional byte-offset syntax and new element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to buffer in UB memory space (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector register |

**Constraints**:
- Buffer must be in UB memory space
- For byte-offset syntax: offset must be properly aligned based on element type
- For element-indexing syntax: the requested vector region must be within tile bounds and satisfy alignment requirements

**Examples**:
```python
# Traditional byte-offset syntax
vec = pto.vlds(ub_ptr, lane * 256)

# New element-indexing syntax
vec = pto.vlds(tile[i, j:])      # Load from row i, columns j to j+vector_lanes-1
vec = pto.vlds(tile[k:])         # Load from 1D tile, elements k to k+vector_lanes-1

# Generic kernel that works for both f16 and f32
@pto.vkernel(target="a5", op="scale", dtypes=[(pto.AnyFloat, pto.AnyFloat)], priority=10)
def generic_scale(src: pto.Tile, dst: pto.Tile, scale: pto.f32):
    rows, cols = src.shape
    all_mask = pto.make_mask(src.element_type, PAT.ALL)
    for i in range(0, rows):
        for j in range(0, cols, vector_lanes):  # vector_lanes computed from element type
            # No manual byte calculation needed!
            vec = pto.vlds(src[i, j:])
            scaled = pto.vmuls(vec, scale, all_mask)
            pto.vsts(scaled, dst[i, j:], all_mask)
```

#### `pto.vldas(buf: ptr, offset: Index, align: pto.align) -> VRegType`  [Advanced Tier]
#### `pto.vldas(tile[row, col:], align: pto.align) -> VRegType`  [Basic Tier]
#### `pto.vldas(tile[start:], align: pto.align) -> VRegType`  [Basic Tier]

**Description**: Aligned vector load with explicit alignment carrier. Supports both byte-offset and element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to buffer in UB memory space (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |
| `align` | `pto.align` | Alignment specification |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index |
| `align` | `pto.align` | Alignment specification |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector register |

**Examples**:
```python
# Byte-offset syntax
vec = pto.vldas(ub_ptr, offset, align)

# Element-indexing syntax
vec = pto.vldas(tile[i, j:], align)
vec = pto.vldas(tile[k:], align)
```

#### `pto.vldus(buf: ptr, offset: Index) -> VRegType`  [Advanced Tier]
#### `pto.vldus(tile[row, col:]) -> VRegType`  [Basic Tier]
#### `pto.vldus(tile[start:]) -> VRegType`  [Basic Tier]

**Description**: Unaligned vector load. Supports both byte-offset and element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to buffer in UB memory space (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector register |

**Examples**:
```python
# Byte-offset syntax
vec = pto.vldus(ub_ptr, offset)

# Element-indexing syntax
vec = pto.vldus(tile[i, j:])
vec = pto.vldus(tile[k:])
```

#### `pto.vplds(buf: ptr, offset: Index, pred: MaskType) -> VRegType`  [Advanced Tier]
#### `pto.vplds(tile[row, col:], pred: MaskType) -> VRegType`  [Basic Tier]
#### `pto.vplds(tile[start:], pred: MaskType) -> VRegType`  [Basic Tier]

**Description**: Predicated vector load stateless. Supports both byte-offset and element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to buffer in UB memory space (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |
| `pred` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index |
| `pred` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector register |

**Examples**:
```python
# Byte-offset syntax
vec = pto.vplds(ub_ptr, offset, mask)

# Element-indexing syntax
vec = pto.vplds(tile[i, j:], mask)
vec = pto.vplds(tile[k:], mask)
```

#### `pto.vldx2(buf1: ptr, buf2: ptr, offset: Index) -> (VRegType, VRegType)`  [Advanced Tier]
#### `pto.vldx2(tile1[row, col:], tile2[row, col:]) -> (VRegType, VRegType)`  [Basic Tier]
#### `pto.vldx2(tile1[start:], tile2[start:]) -> (VRegType, VRegType)`  [Basic Tier]

**Description**: Dual vector load from two buffers. Supports both byte-offset and element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf1` | `ptr` | Pointer to first buffer (Advanced mode only - requires explicit pointer) |
| `buf2` | `ptr` | Pointer to second buffer (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset (applied to both buffers) |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile1[row, col:]` | `Tile` with indexing | First 2D tile with row index and starting column |
| `tile2[row, col:]` | `Tile` with indexing | Second 2D tile with row index and starting column |
| _or_ | | |
| `tile1[start:]` | `Tile` with indexing | First 1D tile with starting element index |
| `tile2[start:]` | `Tile` with indexing | Second 1D tile with starting element index |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec1` | `VRegType` | Vector from first buffer |
| `vec2` | `VRegType` | Vector from second buffer |

**Examples**:
```python
# Byte-offset syntax
vec1, vec2 = pto.vldx2(ub_ptr1, ub_ptr2, offset)

# Element-indexing syntax
vec1, vec2 = pto.vldx2(tile_a[i, j:], tile_b[i, j:])
vec1, vec2 = pto.vldx2(tile_a[k:], tile_b[k:])
```

#### `pto.vsld(buf: ptr, offset: Index) -> VRegType`  [Advanced Tier]
#### `pto.vsld(tile[row, col]) -> VRegType`  [Basic Tier]
#### `pto.vsld(tile[pos]) -> VRegType`  [Basic Tier]

**Description**: Scalar load to vector (broadcast scalar to all lanes). Supports both byte-offset and element-indexing syntax. The element-indexing syntax loads a single element (not a vector) and broadcasts it to all lanes.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to buffer in UB memory space (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col]` | `Tile` with indexing | 2D tile with row and column indices (single element) |
| `tile[pos]` | `Tile` with indexing | 1D tile with element index (single element) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Vector with scalar broadcast to all lanes |

**Examples**:
```python
# Byte-offset syntax
vec = pto.vsld(ub_ptr, offset)

# Element-indexing syntax
vec = pto.vsld(tile[i, j])    # Load single element at (i,j) and broadcast
vec = pto.vsld(tile[k])       # Load single element at position k and broadcast
```

#### `pto.vgather2(buf1: ptr, buf2: ptr, offsets1: Index, offsets2: Index) -> (VRegType, VRegType)`  [Advanced Tier]

**Description**: Dual‑lane gather load from two buffers using irregular access patterns.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf1` | `ptr` | Pointer to first buffer |
| `buf2` | `ptr` | Pointer to second buffer |
| `offsets1` | `Index` | Byte offsets for first buffer |
| `offsets2` | `Index` | Byte offsets for second buffer |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec1` | `VRegType` | Gathered vector from first buffer |
| `vec2` | `VRegType` | Gathered vector from second buffer |

**Example**:
```python
vec1, vec2 = pto.vgather2(buf1, buf2, offsets1, offsets2)
```

#### `pto.vgather2_bc(buf1: ptr, buf2: ptr, offsets1: Index, offsets2: Index, broadcast: ScalarType) -> (VRegType, VRegType)`  [Advanced Tier]

**Description**: Dual‑lane gather load with broadcast scalar to all lanes.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf1` | `ptr` | Pointer to first buffer |
| `buf2` | `ptr` | Pointer to second buffer |
| `offsets1` | `Index` | Byte offsets for first buffer |
| `offsets2` | `Index` | Byte offsets for second buffer |
| `broadcast` | `ScalarType` | Scalar value broadcast to all lanes |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec1` | `VRegType` | Gathered vector from first buffer |
| `vec2` | `VRegType` | Gathered vector from second buffer |

**Example**:
```python
vec1, vec2 = pto.vgather2_bc(buf1, buf2, offsets1, offsets2, pto.f32(1.0))
```

#### `pto.vgatherb(buf: ptr, offsets: Index) -> VRegType`  [Advanced Tier]

**Description**: Byte‑granularity gather load.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to buffer |
| `offsets` | `Index` | Byte offsets |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Gathered vector |

**Example**:
```python
vec = pto.vgatherb(buf, offsets)
```

#### `pto.vsldb(buf: ptr, offset: Index, broadcast: ScalarType) -> VRegType`  [Advanced Tier]
#### `pto.vsldb(tile[row, col], broadcast: ScalarType) -> VRegType`  [Basic Tier]
#### `pto.vsldb(tile[pos], broadcast: ScalarType) -> VRegType`  [Basic Tier]

**Description**: Scalar load with broadcast (enhanced version of `vsld`). Supports both byte‑offset and element‑indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Pointer to buffer |
| `offset` | `Index` | Byte offset |
| `broadcast` | `ScalarType` | Scalar value broadcast to all lanes |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col]` | `Tile` with indexing | 2D tile with row and column indices (single element) |
| `tile[pos]` | `Tile` with indexing | 1D tile with element index (single element) |
| `broadcast` | `ScalarType` | Scalar value broadcast to all lanes |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Vector with loaded scalar broadcast to all lanes |

**Example**:
```python
# Byte-offset syntax
vec = pto.vsldb(ub_ptr, offset, pto.f32(0.0))

# Element-indexing syntax
vec = pto.vsldb(tile[i, j], pto.f32(1.0))
```

### Vector Store Operations

Operations for storing data from vector registers to memory.

#### `pto.vsts(vec: VRegType, buf: ptr, offset: Index, mask: MaskType) -> None`  [Advanced Tier]
#### `pto.vsts(vec: VRegType, tile[row, col:], mask: MaskType) -> None`  [Basic Tier]
#### `pto.vsts(vec: VRegType, tile[start:], mask: MaskType) -> None`  [Basic Tier]

**Description**: Stateless vector store to buffer. Supports both byte-offset and element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `buf` | `ptr` | Pointer to destination buffer in UB memory space (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

**Constraints**:
- Buffer must be in UB memory space
- For byte-offset syntax: offset must be properly aligned based on element type
- For element-indexing syntax: the destination vector region must be within tile bounds and satisfy alignment requirements

**Examples**:
```python
# Byte-offset syntax
pto.vsts(vec_f32, ub_ptr, lane * 256, mask32)

# Element-indexing syntax
pto.vsts(vec, tile[i, j:], mask)      # Store to row i, columns j to j+vector_lanes-1
pto.vsts(vec, tile[k:], mask)         # Store to 1D tile, elements k to k+vector_lanes-1

# In a generic kernel
@pto.vkernel(target="a5", op="copy", dtypes=[(pto.AnyFloat, pto.AnyFloat)], priority=10)
def generic_store(src: pto.Tile, dst: pto.Tile):
    rows, cols = src.shape
    all_mask = pto.make_mask(src.element_type, PAT.ALL)
    for i in range(0, rows):
        for j in range(0, cols, vector_lanes):
            vec = pto.vlds(src[i, j:])
            pto.vsts(vec, dst[i, j:], all_mask)  # No manual offset calculation
```

#### `pto.psts(mask: MaskType, buf: ptr, offset: Index) -> None`  [Advanced Tier]
#### `pto.psts(mask: MaskType, tile[row, col:]) -> None`  
#### `pto.psts(mask: MaskType, tile[start:]) -> None`

**Description**: Predicate store to buffer. Supports both traditional byte-offset syntax and new element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Mask to store |
| `buf` | `ptr` | Pointer to destination buffer (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Mask to store |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |

**Parameters (1D element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Mask to store |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |

**Returns**: None (side-effect operation)

#### `pto.vsst(scalar: ScalarType, buf: ptr, offset: Index, mask: MaskType) -> None`  [Advanced Tier]
#### `pto.vsst(scalar: ScalarType, tile[row, col:], mask: MaskType) -> None`  
#### `pto.vsst(scalar: ScalarType, tile[start:], mask: MaskType) -> None`

**Description**: Scalar to vector store (broadcast scalar to all lanes). Supports both traditional byte-offset syntax and new element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `buf` | `ptr` | Pointer to destination buffer (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| `mask` | `MaskType` | Predicate mask |

**Parameters (1D element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

#### `pto.vstx2(vec1: VRegType, vec2: VRegType, buf1: ptr, buf2: ptr, offset: Index, mask: MaskType) -> None`  [Advanced Tier]
#### `pto.vstx2(vec1: VRegType, vec2: VRegType, tile1[row, col:], tile2[row, col:], mask: MaskType) -> None`  
#### `pto.vstx2(vec1: VRegType, vec2: VRegType, tile1[start:], tile2[start:], mask: MaskType) -> None`

**Description**: Dual vector store to two buffers. Supports both traditional byte-offset syntax and new element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First vector to store |
| `vec2` | `VRegType` | Second vector to store |
| `buf1` | `ptr` | First destination buffer |
| `buf2` | `ptr` | Second destination buffer |
| `offset` | `Index` | Byte offset (applied to both buffers) |
| `mask` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First vector to store |
| `vec2` | `VRegType` | Second vector to store |
| `tile1[row, col:]` | `Tile` with indexing | First 2D tile with row index and starting column (vector-width range) |
| `tile2[row, col:]` | `Tile` with indexing | Second 2D tile with row index and starting column (vector-width range) |
| `mask` | `MaskType` | Predicate mask |

**Parameters (1D element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First vector to store |
| `vec2` | `VRegType` | Second vector to store |
| `tile1[start:]` | `Tile` with indexing | First 1D tile with starting element index (vector-width range) |
| `tile2[start:]` | `Tile` with indexing | Second 1D tile with starting element index (vector-width range) |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

#### `pto.vsta(vec: VRegType, buf: ptr, offset: Index, align: pto.align, mask: MaskType) -> None`  [Advanced Tier]
#### `pto.vsta(vec: VRegType, tile[row, col:], align: pto.align, mask: MaskType) -> None`  
#### `pto.vsta(vec: VRegType, tile[start:], align: pto.align, mask: MaskType) -> None`

**Description**: Aligned vector store with explicit alignment carrier. Supports both traditional byte-offset syntax and new element-indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `buf` | `ptr` | Pointer to destination buffer (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |
| `align` | `pto.align` | Alignment specification |
| `mask` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| `align` | `pto.align` | Alignment specification |
| `mask` | `MaskType` | Predicate mask |

**Parameters (1D element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |
| `align` | `pto.align` | Alignment specification |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

#### `pto.vscatter(vec: VRegType, buf: ptr, offsets: Index, mask: MaskType) -> None`  [Advanced Tier]

**Description**: Scatter store with irregular access pattern.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to scatter |
| `buf` | `ptr` | Pointer to destination buffer |
| `offsets` | `Index` | Byte offsets for scatter locations |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.vscatter(vec, buf, offsets, mask)
```

#### `pto.vsstb(scalar: ScalarType, buf: ptr, offset: Index, mask: MaskType) -> None`  [Advanced Tier]
#### `pto.vsstb(scalar: ScalarType, tile[row, col:], mask: MaskType) -> None`  [Basic Tier]
#### `pto.vsstb(scalar: ScalarType, tile[start:], mask: MaskType) -> None`  [Basic Tier]

**Description**: Scalar to vector store with broadcast (enhanced version of `vsst`). Supports both byte‑offset and element‑indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `buf` | `ptr` | Pointer to destination buffer |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| `mask` | `MaskType` | Predicate mask |

**Parameters (1D element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

**Example**:
```python
# Byte-offset syntax
pto.vsstb(pto.f32(0.0), ub_ptr, offset, mask)

# Element-indexing syntax
pto.vsstb(pto.f32(1.0), tile[i, j:], mask)
```

#### `pto.vstar(vec: VRegType, buf: ptr, offset: Index, align: pto.align, mask: MaskType) -> None`  [Advanced Tier]
#### `pto.vstar(vec: VRegType, tile[row, col:], align: pto.align, mask: MaskType) -> None`  [Basic Tier]
#### `pto.vstar(vec: VRegType, tile[start:], align: pto.align, mask: MaskType) -> None`  [Basic Tier]

**Description**: Vector store with alignment requirement (enhanced version of `vsta`). Supports both byte‑offset and element‑indexing syntax.

**Parameters (byte-offset syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `buf` | `ptr` | Pointer to destination buffer |
| `offset` | `Index` | Byte offset |
| `align` | `pto.align` | Alignment specification |
| `mask` | `MaskType` | Predicate mask |

**Parameters (element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `tile[row, col:]` | `Tile` with indexing | 2D tile with row index and starting column (vector-width range) |
| `align` | `pto.align` | Alignment specification |
| `mask` | `MaskType` | Predicate mask |

**Parameters (1D element-indexing syntax)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `tile[start:]` | `Tile` with indexing | 1D tile with starting element index (vector-width range) |
| `align` | `pto.align` | Alignment specification |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

**Example**:
```python
# Byte-offset syntax
pto.vstar(vec, ub_ptr, offset, align, mask)

# Element-indexing syntax
pto.vstar(vec, tile[i, j:], align, mask)
```

#### `pto.vstas(vec: VRegType, buf: ptr, offset: Index, align: pto.align, mask: MaskType) -> None`  [Advanced Tier]
#### `pto.vstas(vec: VRegType, tile[row, col:], align: pto.align, mask: MaskType) -> None`  [Basic Tier]
#### `pto.vstas(vec: VRegType, tile[start:], align: pto.align, mask: MaskType) -> None`  [Basic Tier]

**Description**: Aligned vector store (alias for `vstar`). Supports both byte‑offset and element‑indexing syntax.

**Parameters**: Same as `pto.vstar`.

**Returns**: None (side-effect operation)

**Example**:
```python
pto.vstas(vec, tile[i, j:], align, mask)
```

### Stateful Store Operations

Operations for storing data with stateful semantics.

#### `pto.pstu(mask: MaskType, buf: ptr, offset: Index) -> None`[Advanced Tier]

**Description**: Predicate stateful store.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Mask to store |
| `buf` | `ptr` | Pointer to destination buffer (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |

**Returns**: None (side-effect operation)

#### `pto.vstu(vec: VRegType, buf: ptr, offset: Index, mask: MaskType) -> None`  [Advanced Tier]

**Description**: Vector stateful store.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `buf` | `ptr` | Pointer to destination buffer (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

#### `pto.vstus(vec: VRegType, buf: ptr, offset: Index, mask: MaskType) -> None`  [Advanced Tier]

**Description**: Vector store update stateless.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `buf` | `ptr` | Pointer to destination buffer (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

#### `pto.vstur(vec: VRegType, buf: ptr, offset: Index, mask: MaskType) -> None`  [Advanced Tier]

**Description**: Vector store update register.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `buf` | `ptr` | Pointer to destination buffer (Advanced mode only - requires explicit pointer) |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)