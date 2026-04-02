# TileLang Python DSL Guide

The TileLang Python DSL provides a high-level, Pythonic interface for authoring vector compute kernels targeting the Ascend NPU hardware. This guide is intended for library developers and performance engineers who need to write efficient, hardware-aware kernels using the PTO micro instruction set.

The DSL is designed to generate MLIR function libraries rather than direct binary executables. These MLIR libraries are intended to be consumed by other compilation frameworks that transform high-level tile semantics into low-level vector operations. This enables library developers to focus on hardware-aware kernel authoring while relying on upstream compilers for tile-level optimizations and code generation.

## Quick Start

**Note on mask pattern enums**: For brevity, examples in this guide use `PAT` as an alias for `pto.MaskPattern` (e.g., `PAT.ALL` instead of `pto.MaskPattern.PAT_ALL`). You can create this alias with `from pto import MaskPattern as PAT` or `PAT = pto.MaskPattern`.

Here's a minimal example of a tile scaling kernel using the new Tile type:

```python
import pto

@pto.vkernel(target="a5", name="tile_scale")
def tile_scale(input_tile: pto.Tile,   # Input tile (shape: 256x128, f32, GM)
               output_tile: pto.Tile,  # Output tile (same shape and type)
               scale_factor: pto.f32): # Scaling factor
    # Access tile properties
    rows, cols = input_tile.shape      # (256, 128)
    dtype = input_tile.element_type    # pto.f32
    
    # Create a temporary tile in UB for computation
    ub_tile = pto.tile((rows, cols), dtype, pto.MemorySpace.UB)
    
    # Copy input tile from GM to UB
    # Note: Simplified syntax pto.copy_gm_to_ub would be provided as syntax sugar
    # Using underlying operation with explicit parameters
    pto.copy_gm_to_ubuf(input_tile.as_ptr(), ub_tile.as_ptr(),
                        0, cols * 4, cols * 4,  # strides in bytes
                        0, 0, False, 0, cols * 4, cols * 4)
    
    # Vector computation: scale all elements in the tile
    with pto.vecscope():
        all_mask = pto.make_mask(dtype, PAT.ALL)
        
        # Process tile in row-major order
        for row in range(0, rows):
            row_offset = row * cols * 4  # f32 = 4 bytes
            
            # Process each row in vector chunks (64 elements per vector)
            for chunk in range(0, cols, 64):
                offset = row_offset + chunk * 4
                
                # Load vector from UB tile (implicit tile→UBRef conversion)
                vec = pto.vlds(ub_tile, offset)
                
                # Scale vector
                scaled = pto.vmuls(vec, scale_factor, all_mask)
                
                # Store result back to UB tile
                pto.vsts(scaled, ub_tile, offset, all_mask)
    
    # Copy result from UB back to GM output tile
    # Note: Simplified syntax pto.copy_ub_to_gm would be provided as syntax sugar
    pto.copy_ubuf_to_gm(ub_tile.as_ptr(), output_tile.as_ptr(),
                        0, cols * 4, cols * 4,
                        0, cols * 4, cols * 4)
```

This example demonstrates:
1. **Tile parameters** in kernel declaration
2. **Tile property access** (shape, element_type)  
3. **Tile creation** for temporary buffers
4. **Tile to pointer conversion** for copy operations (`.as_ptr()`)
5. **Implicit tile→UBRef conversion** in vector load/store operations
6. **Future syntax sugar** for simplified copy operations (commented)

For an even more concise example showing pure computation on UB tiles (assuming data is already in UB):

```python
@pto.vkernel(target="a5", name="ub_tile_computation")
def ub_tile_computation(a: pto.Tile,  # UB tile
                        b: pto.Tile,  # UB tile  
                        c: pto.Tile): # UB tile (output)
    dtype = a.element_type

    # All tiles are in UB memory space
    with pto.vecscope():
        all_mask = pto.make_mask(dtype, PAT.ALL)
        rows, cols = a.shape
        
        # Element-wise: c = a + b * 2.0
        for i in range(0, rows * cols, 64):
            # Load vectors from UB tiles
            vec_a = pto.vlds(a, i * 4)      # Implicit tile→UBRef
            vec_b = pto.vlds(b, i * 4)
            
            # Compute: b * 2.0
            scaled_b = pto.vmuls(vec_b, 2.0, all_mask)
            
            # Compute: a + scaled_b
            result = pto.vadd(vec_a, scaled_b, all_mask)
            
            # Store result to output tile
            pto.vsts(result, c, i * 4, all_mask)
```

## Core Concepts

### Kernel Declaration

Kernels are defined using the `@pto.vkernel` decorator:

```python
@pto.vkernel(target="a5", name="kernel_name")
def kernel_name(param1: type1, param2: type2) -> None:
    # kernel body
```

- `target`: Target architecture (currently "a5" for Ascend 950)
- `name`: Optional kernel name (defaults to function name)
- Parameters must have type annotations using DSL types

### Value Model

The DSL operates on symbolic values, not Python runtime values:
- **Constants**: Python literals that are typed to machine types
- **Operation results**: Values produced by DSL operations
- **Block arguments**: Values introduced by control flow structures

### Memory Spaces

The DSL supports different memory spaces:
- `MemorySpace.GM`: Global Memory
- `MemorySpace.UB`: Unified Buffer (local storage for vector computation)

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

### Pointer Types

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

### Pointer Type Aliases

For clarity in API documentation, the following type aliases are used:

| Alias | Equivalent Type | Description |
|-------|----------------|-------------|
| `GMPtr` | `ptr(..., MemorySpace.GM)` | Pointer to Global Memory |
| `UBPtr` | `ptr(..., MemorySpace.UB)` | Pointer to Unified Buffer |
| `UBRef` | `Union[MemRefType, UBPtr]` | UB buffer or pointer (accepted by load/store ops) |
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

### Alignment Type

The `pto.align` type is used for alignment carrier operations and maps to `!pto.align`.

### Tile Types

Tile types represent data blocks in memory with layout and configuration information, corresponding to `!pto.tile_buf` in the VPTO IR. Tiles are commonly used as kernel parameters for tiled computations.

#### Tile Type Definition

```python
# Create a tile with shape, element type, and memory space
tile = pto.tile((256, 128), pto.f32, MemorySpace.UB)

# With explicit configuration
config = pto.tile_config(
    b_layout=pto.BLayout.ROW_MAJOR,
    s_layout=pto.SLayout.NONE_BOX,
    s_fractal_size=pto.i32(16),
    pad_value=pto.PadValue.ZERO
)
tile = pto.tile((256, 128), pto.f32, MemorySpace.UB, config=config)

# With valid shape (actual data dimensions within tile)
tile = pto.tile((256, 128), pto.f32, MemorySpace.UB, valid_shape=(240, 120))
```

**Important Notes on Shape and Valid Shape:**
- **Static Shape Requirement**: The `shape` parameter must be a compile-time constant. Tile dimensions are fixed at compilation time and cannot change at runtime.
- **Valid Shape Constraints**: The `valid_shape` parameter can be either static (compile-time constant) or dynamic (determined at runtime). It must be less than or equal to the physical `shape` in each dimension. This allows for variable-sized data within a fixed tile allocation.
- **Default Behavior**: When `valid_shape` is not specified, it defaults to the full `shape`.

#### Tile Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | **Static** full tile dimensions (compile-time constant) |
| `element_type` | `Type` | Element data type (e.g., `pto.f32`) |
| `memory_space` | `MemorySpace` | Memory space (GM, UB, etc.) |
| `valid_shape` | `tuple[int, ...]` | Actual data dimensions within tile (can be static/compile-time or dynamic/runtime). Must be ≤ shape in each dimension. |
| `config` | `TileConfig` | Layout and padding configuration |

#### Tile Configuration

The tile configuration includes layout and padding information:

```python
# Layout enums
pto.BLayout.ROW_MAJOR     # 0: row-major base layout
pto.BLayout.COL_MAJOR     # 1: column-major base layout

pto.SLayout.NONE_BOX      # 0: no secondary layout
pto.SLayout.ROW_MAJOR     # 1: row-major secondary layout  
pto.SLayout.COL_MAJOR     # 2: column-major secondary layout

pto.PadValue.NULL         # 0: no padding
pto.PadValue.ZERO         # 1: zero padding
pto.PadValue.MAX          # 2: maximum value padding
pto.PadValue.MIN          # 3: minimum value padding
```

#### Tile Shape Concepts

- **Static Physical Shape**: The `shape` parameter represents the **static physical dimensions** of the tile allocated in memory. This must be a **compile-time constant** because tile memory allocation is fixed during compilation. The shape determines the total memory footprint and cannot change at runtime.

- **Valid Shape**: The `valid_shape` parameter represents the logical dimensions of actual data within the tile. It can be either **static** (compile-time constant) or **dynamic** (determined at runtime). It must be less than or equal to the physical `shape` in each dimension. When `valid_shape` is not specified, it defaults to the full `shape`.

- **Key Distinction**:
  - `shape`: **Static, compile-time** - Fixed tile allocation
  - `valid_shape`: **Static or Dynamic** - Actual data region (must be ≤ shape)

- **Constraints**:
  - `valid_shape[i] ≤ shape[i]` for each dimension i
  - `shape` must be compile-time constants
  - `valid_shape` can be compile-time constants or runtime values

- **Use Cases**:
  - Fixed-size tile buffers with variable data (e.g., batch processing with different input sizes)
  - Padding scenarios where physical allocation is larger than actual data
  - Partial tile utilization in tiled algorithms

- **Fractal Layout**: The `s_fractal_size` in tile configuration specifies the size of fractal blocks for secondary layout. This is used for optimized memory access patterns in matrix operations.

- **Padding Behavior**: The `pad_value` determines how out-of-bounds accesses are handled when reading beyond `valid_shape` but within `shape`. Padding values are used for accesses in the padded region (between valid_shape and shape).

> **⚠️ Important: Shape Constraints**
> 
> The tile `shape` must be **compile-time constants**. `valid_shape` can be compile-time constants or determined at runtime, but must satisfy `valid_shape[i] ≤ shape[i]` for all dimensions i.

### Tile Operations

#### Basic Access Operations

```python
# Get tile properties
shape = tile.shape                    # (256, 128)
elem_type = tile.element_type         # pto.f32
mem_space = tile.memory_space         # MemorySpace.UB
valid_shape = tile.valid_shape        # (240, 120) or same as shape

# Get configuration properties
config = tile.config
b_layout = config.b_layout            # pto.BLayout.ROW_MAJOR
s_layout = config.s_layout            # pto.SLayout.NONE_BOX
s_fractal = config.s_fractal_size     # pto.i32(16)
pad = config.pad_value                # pto.PadValue.ZERO

# Dynamic properties
rank = tile.rank                      # 2
num_elements = tile.num_elements      # 32768 (256 * 128)
valid_elements = tile.valid_elements  # 28800 (240 * 120)
```

#### Layout and Stride Queries

```python
# Get layout descriptors
layout_desc = tile.layout_descriptor  # Returns layout description object

# Get strides (in elements)
strides = tile.strides                # (128, 1) for row-major 256x128

# Get byte strides
byte_strides = tile.byte_strides      # (512, 4) for f32 row-major

# Get base offset (in bytes)
offset = tile.offset                  # pto.i64(0) or specified offset
```

#### Conversion Operations

Tiles support both explicit and implicit conversion to UBRef. When a tile is used in operations expecting a UBRef (e.g., `pto.vlds`, `pto.vsts`), it is automatically converted.

```python
# Convert to UBRef (implicit in vector operations)
ub_ref = tile.to_ubref()              # Explicit conversion
# or use tile as UBRef directly in vector ops
vec = pto.vlds(tile, offset)          # Implicit conversion

# Convert to typed pointer
ptr = tile.as_ptr()                   # Returns pto.ptr(pto.f32, MemorySpace.UB)

# Convert to MemRef (for compatibility)
memref = tile.to_memref()             # Returns pto.memref((256, 128), pto.f32, MemorySpace.UB)

# Extract slice of tile
slice_tile = tile.slice((0, 0), (64, 128))  # 64x128 slice from top-left corner

# Reshape tile (logical reshape, no data movement)
reshaped = tile.reshape((32768,))     # 1D reshape of 256x128 tile
```

#### Kernel Parameter Usage

```python
@pto.vkernel(target="a5", name="tiled_kernel")
def tiled_kernel(
    input_tile: pto.Tile,              # Tile parameter
    output_tile: pto.Tile,             # Another tile parameter
    scale: pto.f32
):
    # Convert tiles to UBRef for vector operations
    ub_in = input_tile.to_ubref()
    ub_out = output_tile.to_ubref()
    
    # Or use tiles directly (implicit conversion)
    with pto.vecscope():
        all_mask = pto.make_mask(pto.f32, PAT.ALL)
        for i in range(0, 256, 64):
            # tile implicitly converts to UBRef in vlds
            vec = pto.vlds(input_tile, i * 128 * 4)  # Byte offset for row i
            scaled = pto.vmuls(vec, scale, all_mask)
            pto.vsts(scaled, output_tile, i * 128 * 4, all_mask)
```

#### Tile Creation from Existing Buffers

```python
# Create tile from existing pointer with shape
ptr = pto.castptr(0, pto.ptr(pto.f32, MemorySpace.UB))
tile = pto.tile_from_ptr(ptr, (256, 128), pto.f32)

# Create tile from memref
memref = pto.memref((256, 128), pto.f32, MemorySpace.UB)
tile = pto.tile_from_memref(memref)

# Create tile with explicit stride
tile = pto.tile_with_strides((256, 128), pto.f32, MemorySpace.UB, 
                             strides=(256, 1))  # Column-major strides
```

## Control Flow

### Vector Scopes

Vector scopes define regions for vector computation:

**Regular vector scope** (implicit capture):
```python
with pto.vecscope():
    # Can reference outer values
    vec = pto.vlds(outer_ptr, offset)
    pto.vsts(vec, dst_ptr, offset, mask)
```

**Strict vector scope** (explicit capture only):
```python
with pto.strict_vecscope(src_ptr, dst_ptr, start, end) as (s, d, lb, ub):
    # Can only use: s, d, lb, ub and locally defined values
    for i in range(lb, ub, 64):
        vec = pto.vlds(s, i)
        pto.vsts(vec, d, i, all_mask)
```

Strict scopes enforce explicit data flow and prevent implicit captures.

### Loops

Counted loops use Python's `range` syntax:

```python
for i in range(lb, ub, step):
    # Loop body
    mask, rem = pto.make_mask(pto.f32, remaining)
    # ...
```

Loop-carried state is automatically handled through variable updates within the loop.

### Conditionals

`if` statements support value merging:

```python
flag: pto.i1 = some_condition
step: pto.i32 = 0

if flag:
    step = pto.i32(64)
else:
    step = pto.i32(128)

# 'step' here is the merged result from both branches
```

Variables defined in only one branch are local to that branch.

## Operations

The DSL provides operations grouped by functionality. All operations use the `pto.` prefix. Operations are organized by functional families following the VPTO instruction set architecture.

### Pointer Construction

Operations for creating and manipulating typed pointers.

#### `pto.castptr(offset: pto.i64, ptr_type: Type) -> PtrType`

**Description**: Creates a pointer with the specified offset and type.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `offset` | `pto.i64` | Byte offset from base address |
| `ptr_type` | `Type` | Target pointer type (e.g., `pto.ptr(pto.f32, MemorySpace.GM)`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `ptr` | `PtrType` | Typed pointer value |

**Example**:
```python
ub_ptr = pto.castptr(0, pto.ptr(pto.f32, MemorySpace.UB))
```

#### `pto.addptr(ptr: PtrType, offset: pto.i64) -> PtrType`

**Description**: Adds an offset to an existing pointer.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `ptr` | `PtrType` | Source pointer |
| `offset` | `pto.i64` | Byte offset to add |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `new_ptr` | `PtrType` | Pointer with offset applied |

**Example**:
```python
next_ptr = pto.addptr(ub_ptr, 4096)
```

### Synchronization & Buffer Control

Operations for pipeline synchronization and buffer management.

#### `pto.set_flag(pipe_from: PIPE, pipe_to: PIPE, event: EVENT) -> None`

**Description**: Sets a synchronization flag between hardware pipelines.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe_from` | `PIPE` | Source pipeline (e.g., `PIPE.MTE2`) |
| `pipe_to` | `PIPE` | Destination pipeline (e.g., `PIPE.V`) |
| `event` | `EVENT` | Event identifier (e.g., `EVENT.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import PIPE, EVENT

pto.set_flag(PIPE.MTE2, PIPE.V, EVENT.ID0)
```

#### `pto.wait_flag(pipe_from: PIPE, pipe_to: PIPE, event: EVENT) -> None`

**Description**: Waits for a synchronization flag between hardware pipelines.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe_from` | `PIPE` | Source pipeline (e.g., `PIPE.MTE2`) |
| `pipe_to` | `PIPE` | Destination pipeline (e.g., `PIPE.V`) |
| `event` | `EVENT` | Event identifier (e.g., `EVENT.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import PIPE, EVENT

pto.wait_flag(PIPE.MTE2, PIPE.V, EVENT.ID0)
```

#### `pto.pipe_barrier(pipes: PIPE) -> None`

**Description**: Executes a barrier across specified pipelines.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipes` | `PIPE` | Pipeline specification (e.g., `PIPE.ALL`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import PIPE

pto.pipe_barrier(PIPE.ALL)
```

#### `pto.get_buf(op_type: SyncOpType, buf_id: pto.i32, mode: pto.i32 = 0) -> None`

**Description**: Acquires a buffer for producer-consumer synchronization.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `op_type` | `SyncOpType` | Operation type (e.g., `SyncOpType.TLOAD`) |
| `buf_id` | `pto.i32` | Buffer identifier |
| `mode` | `pto.i32` | Acquisition mode (default: 0) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import SyncOpType

pto.get_buf(SyncOpType.TLOAD, 0)
```

#### `pto.rls_buf(op_type: SyncOpType, buf_id: pto.i32, mode: pto.i32 = 0) -> None`

**Description**: Releases a previously acquired buffer.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `op_type` | `SyncOpType` | Operation type (e.g., `SyncOpType.TLOAD`) |
| `buf_id` | `pto.i32` | Buffer identifier |
| `mode` | `pto.i32` | Release mode (default: 0) |

**Returns**: None (side-effect操作)

**Example**:
```python
from pto import SyncOpType

pto.rls_buf(SyncOpType.TLOAD, 0)
```

### Copy Programming

Operations for configuring DMA transfer parameters.

#### `pto.set_loop2_stride_outtoub(stride0: pto.i64, stride1: pto.i64) -> None`

**Description**: Configures DMA stride parameters for GM → UB transfers (loop2).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `stride0` | `pto.i64` | First dimension stride |
| `stride1` | `pto.i64` | Second dimension stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop1_stride_outtoub(stride0: pto.i64, stride1: pto.i64) -> None`

**Description**: Configures DMA stride parameters for GM → UB transfers (loop1).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `stride0` | `pto.i64` | First dimension stride |
| `stride1` | `pto.i64` | Second dimension stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop_size_outtoub(size0: pto.i64, size1: pto.i64) -> None`

**Description**: Configures DMA transfer size for GM → UB transfers.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `size0` | `pto.i64` | First dimension size |
| `size1` | `pto.i64` | Second dimension size |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.set_loop_size_outtoub(1, 1)
```

#### `pto.set_loop2_stride_ubtoout(stride0: pto.i64, stride1: pto.i64) -> None`

**Description**: Configures DMA stride parameters for UB → GM transfers (loop2).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `stride0` | `pto.i64` | First dimension stride |
| `stride1` | `pto.i64` | Second dimension stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop1_stride_ubtoout(stride0: pto.i64, stride1: pto.i64) -> None`

**Description**: Configures DMA stride parameters for UB → GM transfers (loop1).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `stride0` | `pto.i64` | First dimension stride |
| `stride1` | `pto.i64` | Second dimension stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop_size_ubtoout(size0: pto.i64, size1: pto.i64) -> None`

**Description**: Configures DMA transfer size for UB → GM transfers.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `size0` | `pto.i64` | First dimension size |
| `size1` | `pto.i64` | Second dimension size |

**Returns**: None (side-effect operation)

### Copy Transfers

Operations for executing DMA data transfers.

#### `pto.copy_gm_to_ubuf(src: GMPtr, dst: UBPtr, src_offset: pto.i64, src_stride0: pto.i64, src_stride1: pto.i64, dst_offset: pto.i64, dst_stride0: pto.i64, transpose: pto.i1, pad_left: pto.i64, pad_right: pto.i64, pad_value: pto.i64) -> None`

**Description**: Copies data from Global Memory (GM) to Unified Buffer (UB).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `GMPtr` | Source GM pointer |
| `dst` | `UBPtr` | Destination UB pointer |
| `src_offset` | `pto.i64` | Source offset |
| `src_stride0` | `pto.i64` | Source stride dimension 0 |
| `src_stride1` | `pto.i64` | Source stride dimension 1 |
| `dst_offset` | `pto.i64` | Destination offset |
| `dst_stride0` | `pto.i64` | Destination stride dimension 0 |
| `transpose` | `pto.i1` | Transpose flag |
| `pad_left` | `pto.i64` | Left padding size |
| `pad_right` | `pto.i64` | Right padding size |
| `pad_value` | `pto.i64` | Padding value |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.copy_gm_to_ubuf(gm_ptr, ub_ptr, 0, 32, 128, 0, 0, False, 0, 128, 128)
```

#### `pto.copy_ubuf_to_ubuf(src: UBPtr, dst: UBPtr, src_offset: pto.i64, src_stride0: pto.i64, src_stride1: pto.i64, dst_offset: pto.i64, dst_stride0: pto.i64, dst_stride1: pto.i64) -> None`

**Description**: Copies data within Unified Buffer (UB → UB).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `UBPtr` | Source UB pointer |
| `dst` | `UBPtr` | Destination UB pointer |
| `src_offset` | `pto.i64` | Source offset |
| `src_stride0` | `pto.i64` | Source stride dimension 0 |
| `src_stride1` | `pto.i64` | Source stride dimension 1 |
| `dst_offset` | `pto.i64` | Destination offset |
| `dst_stride0` | `pto.i64` | Destination stride dimension 0 |
| `dst_stride1` | `pto.i64` | Destination stride dimension 1 |

**Returns**: None (side-effect operation)

#### `pto.copy_ubuf_to_gm(src: UBPtr, dst: GMPtr, src_offset: pto.i64, src_stride0: pto.i64, src_stride1: pto.i64, dst_offset: pto.i64, dst_stride0: pto.i64, dst_stride1: pto.i64) -> None`

**Description**: Copies data from Unified Buffer (UB) to Global Memory (GM).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `UBPtr` | Source UB pointer |
| `dst` | `GMPtr` | Destination GM pointer |
| `src_offset` | `pto.i64` | Source offset |
| `src_stride0` | `pto.i64` | Source stride dimension 0 |
| `src_stride1` | `pto.i64` | Source stride dimension 1 |
| `dst_offset` | `pto.i64` | Destination offset |
| `dst_stride0` | `pto.i64` | Destination stride dimension 0 |
| `dst_stride1` | `pto.i64` | Destination stride dimension 1 |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.copy_ubuf_to_gm(ub_ptr, gm_ptr, 0, 32, 128, 0, 128, 128)
```

### Vector Load Operations

Operations for loading data from memory into vector registers.

#### `pto.vlds(buf: UBRef, offset: Index) -> VRegType`

**Description**: Stateless vector load from buffer.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `UBRef` | Buffer or pointer (UB memory space) |
| `offset` | `Index` | Byte offset |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector register |

**Constraints**:
- Buffer must be in UB memory space
- Offset must be properly aligned based on element type

**Example**:
```python
vec = pto.vlds(ub_ptr, lane * 256)
```

#### `pto.vldas(buf: UBRef, offset: Index, align: pto.align) -> VRegType`

**Description**: Aligned vector load with explicit alignment carrier.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `UBRef` | Buffer or pointer (UB memory space) |
| `offset` | `Index` | Byte offset |
| `align` | `pto.align` | Alignment specification |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector register |

#### `pto.vldus(buf: UBRef, offset: Index) -> VRegType`

**Description**: Unaligned vector load.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `UBRef` | Buffer or pointer (UB memory space) |
| `offset` | `Index` | Byte offset |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector register |

#### `pto.vplds(buf: UBRef, offset: Index, pred: MaskType) -> VRegType`

**Description**: Predicated vector load stateless.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `UBRef` | Buffer or pointer (UB memory space) |
| `offset` | `Index` | Byte offset |
| `pred` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector register |

#### `pto.vldx2(buf1: UBRef, buf2: UBRef, offset: Index) -> (VRegType, VRegType)`

**Description**: Dual vector load from two buffers.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf1` | `UBRef` | First buffer or pointer |
| `buf2` | `UBRef` | Second buffer or pointer |
| `offset` | `Index` | Byte offset (applied to both buffers) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec1` | `VRegType` | Vector from first buffer |
| `vec2` | `VRegType` | Vector from second buffer |

#### `pto.vsld(buf: UBRef, offset: Index) -> VRegType`

**Description**: Scalar load to vector (broadcast scalar to all lanes).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `UBRef` | Buffer or pointer (UB memory space) |
| `offset` | `Index` | Byte offset |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Vector with scalar broadcast to all lanes |

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

#### `pto.ppack(mask: MaskType) -> pto.i32`

**Description**: Packs mask bits into a 32-bit integer.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Input mask (`mask_b8`, `mask_b16`, or `mask_b32`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `packed` | `pto.i32` | Packed mask bits |

#### `pto.punpack(packed: pto.i32) -> MaskType`

**Description**: Unpacks 32-bit integer to mask (granularity determined by context).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `packed` | `pto.i32` | Packed mask bits |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `MaskType` | Unpacked mask |

#### `pto.pnot(mask: MaskType) -> MaskType`

**Description**: Logical negation of mask bits.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Input mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `negated` | `MaskType` | Negated mask |

#### `pto.psel(mask: MaskType, true_val: ScalarType, false_val: ScalarType) -> ScalarType`

**Description**: Selects between two scalar values based on mask.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Selection mask |
| `true_val` | `ScalarType` | Value selected when mask bit is 1 |
| `false_val` | `ScalarType` | Value selected when mask bit is 0 |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `ScalarType` | Selected scalar value |

### Unary Vector Operations

Element-wise unary operations on vector registers.

#### `pto.vabs(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Absolute value of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask (granularity must match vector element type) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Absolute values |

**Constraints**:
- Mask granularity must match vector element type (e.g., `f32` requires `mask_b32`)

**Example**:
```python
abs_vec = pto.vabs(vec_f32, mask32)
```

#### `pto.vexp(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Exponential of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Exponential values |

#### `pto.vln(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Natural logarithm of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Natural logarithm values |

#### `pto.vsqrt(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Square root of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Square root values |

#### `pto.vrec(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Reciprocal of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Reciprocal values |

#### `pto.vrelu(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: ReLU activation (max(0, x)) of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | ReLU-activated values |

#### `pto.vnot(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Bitwise NOT of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bitwise NOT values |

#### `pto.vcadd(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Complex addition of vector elements (treating pairs as complex numbers).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (interpreted as complex pairs) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Complex addition result |

#### `pto.vcmax(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Complex maximum of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector (interpreted as complex pairs) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Complex maximum result |

#### `pto.vbcnt(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Bit count (population count) of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bit count values |

### Binary Vector Operations

Element-wise binary operations on vector registers.

#### `pto.vadd(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise addition of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Sum of vectors |

**Example**:
```python
sum_vec = pto.vadd(vec_a, vec_b, mask32)
```

#### `pto.vsub(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise subtraction of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Difference of vectors |

#### `pto.vmul(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise multiplication of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Product of vectors |

#### `pto.vdiv(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise division of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Quotient of vectors |

#### `pto.vmax(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise maximum of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Element-wise maximum |

#### `pto.vmin(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise minimum of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Element-wise minimum |

#### `pto.vand(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise bitwise AND of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bitwise AND result |

#### `pto.vor(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise bitwise OR of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bitwise OR result |

#### `pto.vxor(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise bitwise XOR of two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Bitwise XOR result |

#### `pto.vshl(vec: VRegType, shift: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise shift left (vector shift amounts).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `shift` | `VRegType` | Shift amounts (per element) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Shifted values |

#### `pto.vshr(vec: VRegType, shift: VRegType, mask: MaskType) -> VRegType`

**Description**: Element-wise shift right (vector shift amounts).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `shift` | `VRegType` | Shift amounts (per element) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Shifted values |

### Vector-Scalar Operations

Operations between vectors and scalars.

#### `pto.vmuls(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Vector multiplied by scalar (broadcast).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar multiplier |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Scaled vector |

**Example**:
```python
scaled = pto.vmuls(vec_f32, pto.f32(2.0), mask32)
```

#### `pto.vadds(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Vector plus scalar (broadcast).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar addend |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result vector |

#### `pto.vmaxs(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Element-wise maximum of vector and scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar value |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Maximum values |

#### `pto.vmins(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Element-wise minimum of vector and scalar.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar value |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Minimum values |

#### `pto.vlrelu(vec: VRegType, scalar: ScalarType, mask: MaskType) -> VRegType`

**Description**: Leaky ReLU activation (max(αx, x)).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Alpha coefficient |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Leaky ReLU activated values |

#### `pto.vshls(vec: VRegType, shift: ScalarType, mask: MaskType) -> VRegType`

**Description**: Vector shift left by scalar (uniform shift).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `shift` | `ScalarType` | Shift amount (same for all elements) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Shifted values |

#### `pto.vshrs(vec: VRegType, shift: ScalarType, mask: MaskType) -> VRegType`

**Description**: Vector shift right by scalar (uniform shift).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `shift` | `ScalarType` | Shift amount (same for all elements) |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Shifted values |

### Carry & Select Operations

Operations with carry propagation and selection.

#### `pto.vaddc(vec1: VRegType, vec2: VRegType, carry_in: ScalarType, mask: MaskType) -> (VRegType, ScalarType)`

**Description**: Vector addition with carry input and output.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `carry_in` | `ScalarType` | Input carry bit |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Sum vector |
| `carry_out` | `ScalarType` | Output carry bit |

#### `pto.vsubc(vec1: VRegType, vec2: VRegType, borrow_in: ScalarType, mask: MaskType) -> (VRegType, ScalarType)`

**Description**: Vector subtraction with borrow input and output.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `borrow_in` | `ScalarType` | Input borrow bit |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Difference vector |
| `borrow_out` | `ScalarType` | Output borrow bit |

#### `pto.vsel(mask: MaskType, true_vec: VRegType, false_vec: VRegType) -> VRegType`

**Description**: Vector select based on mask.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Selection mask |
| `true_vec` | `VRegType` | Vector selected when mask bit is 1 |
| `false_vec` | `VRegType` | Vector selected when mask bit is 0 |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Selected vector |

**Example**:
```python
result = pto.vsel(mask32, scaled_vec, original_vec)
```

### Data Rearrangement

Operations for rearranging data within vectors.

#### `pto.pdintlv_b8(mask: pto.mask_b8) -> pto.mask_b8`

**Description**: Deinterleave 8-bit mask.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `pto.mask_b8` | Input 8-bit mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `pto.mask_b8` | Deinterleaved mask |

#### `pto.pintlv_b16(mask: pto.mask_b16) -> pto.mask_b16`

**Description**: Interleave 16-bit mask.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `pto.mask_b16` | Input 16-bit mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `pto.mask_b16` | Interleaved mask |

#### `pto.vintlv(vec1: VRegType, vec2: VRegType, mask: MaskType) -> VRegType`

**Description**: Interleave two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Interleaved vector |

#### `pto.vdintlv(vec: VRegType, mask: MaskType) -> (VRegType, VRegType)`

**Description**: Deinterleave vector into two vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `vec1` | `VRegType` | First deinterleaved vector |
| `vec2` | `VRegType` | Second deinterleaved vector |

### Conversion & Special Operations

Type conversion and specialized operations.

#### `pto.vtrc(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Truncate vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Truncated vector |

#### `pto.vcvt(vec: VRegType, to_type: Type, mask: MaskType) -> VRegType`

**Description**: Type conversion of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `to_type` | `Type` | Target element type |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Converted vector |

#### `pto.vbitsort(vec: VRegType, mask: MaskType) -> VRegType`

**Description**: Bitonic sort of vector elements.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Sorted vector |

#### `pto.vmrgsort4(vec1: VRegType, vec2: VRegType, vec3: VRegType, vec4: VRegType, mask: MaskType) -> VRegType`

**Description**: 4-way merge sort of vectors.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First input vector |
| `vec2` | `VRegType` | Second input vector |
| `vec3` | `VRegType` | Third input vector |
| `vec4` | `VRegType` | Fourth input vector |
| `mask` | `MaskType` | Predicate mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Merged and sorted vector |

### Stateless Store Operations

Operations for storing data from vector registers to memory (stateless).

#### `pto.vsts(vec: VRegType, buf: UBRef, offset: Index, mask: MaskType) -> None`

**Description**: Stateless vector store to buffer.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `buf` | `UBRef` | Destination buffer or pointer (UB memory space) |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

**Constraints**:
- Buffer must be in UB memory space
- Offset must be properly aligned based on element type

**Example**:
```python
pto.vsts(vec_f32, ub_ptr, lane * 256, mask32)
```

#### `pto.psts(mask: MaskType, buf: UBRef, offset: Index) -> None`

**Description**: Predicate store to buffer.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Mask to store |
| `buf` | `UBRef` | Destination buffer or pointer |
| `offset` | `Index` | Byte offset |

**Returns**: None (side-effect operation)

#### `pto.vsst(scalar: ScalarType, buf: UBRef, offset: Index, mask: MaskType) -> None`

**Description**: Scalar to vector store (broadcast scalar to all lanes).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `ScalarType` | Scalar value |
| `buf` | `UBRef` | Destination buffer or pointer |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

#### `pto.vstx2(vec1: VRegType, vec2: VRegType, buf1: UBRef, buf2: UBRef, offset: Index, mask: MaskType) -> None`

**Description**: Dual vector store to two buffers.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec1` | `VRegType` | First vector to store |
| `vec2` | `VRegType` | Second vector to store |
| `buf1` | `UBRef` | First destination buffer |
| `buf2` | `UBRef` | Second destination buffer |
| `offset` | `Index` | Byte offset (applied to both buffers) |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

#### `pto.vsta(vec: VRegType, buf: UBRef, offset: Index, align: pto.align, mask: MaskType) -> None`

**Description**: Aligned vector store with explicit alignment carrier.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `buf` | `UBRef` | Destination buffer or pointer |
| `offset` | `Index` | Byte offset |
| `align` | `pto.align` | Alignment specification |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

### Stateful Store Operations

Operations for storing data with stateful semantics.

#### `pto.pstu(mask: MaskType, buf: UBRef, offset: Index) -> None`

**Description**: Predicate stateful store.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Mask to store |
| `buf` | `UBRef` | Destination buffer or pointer |
| `offset` | `Index` | Byte offset |

**Returns**: None (side-effect operation)

#### `pto.vstu(vec: VRegType, buf: UBRef, offset: Index, mask: MaskType) -> None`

**Description**: Vector stateful store.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `buf` | `UBRef` | Destination buffer or pointer |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

#### `pto.vstus(vec: VRegType, buf: UBRef, offset: Index, mask: MaskType) -> None`

**Description**: Vector store update stateless.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `buf` | `UBRef` | Destination buffer or pointer |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

#### `pto.vstur(vec: VRegType, buf: UBRef, offset: Index, mask: MaskType) -> None`

**Description**: Vector store update register.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `buf` | `UBRef` | Destination buffer or pointer |
| `offset` | `Index` | Byte offset |
| `mask` | `MaskType` | Predicate mask |

**Returns**: None (side-effect operation)

## Examples

### Simple Vector Copy

```python
@pto.vkernel(name="vector_copy")
def vector_copy(src: pto.memref(256, pto.f32, MemorySpace.UB),
                dst: pto.memref(256, pto.f32, MemorySpace.UB)):
    with pto.vecscope():
        all_mask = pto.make_mask(pto.f32, PAT.ALL)
        for offset in range(0, 256, 64):
            vec = pto.vlds(src, offset)
            pto.vsts(vec, dst, offset, all_mask)
```

### Conditional Computation

```python
@pto.vkernel(name="conditional_scale")
def conditional_scale(src: pto.ptr(pto.f32, MemorySpace.GM),
                      dst: pto.ptr(pto.f32, MemorySpace.GM),
                      threshold: pto.f32):
    # ... setup ...

    with pto.strict_vecscope(ub_in, ub_out, threshold) as (vin, vout, thresh):
        for i in range(0, 1024, 64):
            vec = pto.vlds(vin, i)

            # Compare with threshold
            mask = pto.pge_b32(vec, thresh)

            # Scale values above threshold
            scaled = pto.vmuls(vec, pto.f32(2.0), mask)

            # Keep original values below threshold
            result = pto.vsel(mask, scaled, vec)

            pto.vsts(result, vout, i, all_mask)
```

### Loop with Carry

```python
@pto.vkernel(name="prefix_sum")
def prefix_sum(src: pto.ptr(pto.i32, MemorySpace.UB),
               dst: pto.ptr(pto.i32, MemorySpace.UB)):
    with pto.vecscope():
        all_mask = pto.make_mask(pto.i32, PAT.ALL)
        carry = pto.i32(0)

        for i in range(0, 256, 64):
            vec = pto.vlds(src, i)
            result, carry = pto.vaddcs(vec, carry, all_mask)
            pto.vsts(result, dst, i, all_mask)
```

## Common Errors

### Typed Mask Mismatch

```
Error: f32 vector operation cannot consume mask_b16
```

**Solution:** Ensure mask granularity matches vector element size:
- `f32` vectors use `mask_b32`
- `f16` vectors use `mask_b16`
- `i8` vectors use `mask_b8`

### Strict Scope Implicit Capture

```
Error: strict_vecscope body cannot capture outer value 'ub_in' implicitly
```

**Solution:** Pass all required values in the capture list:

```python
# Wrong:
with pto.strict_vecscope() as ():
    vec = pto.vlds(ub_in, offset)  # ub_in from outer scope

# Correct:
with pto.strict_vecscope(ub_in) as (ub):
    vec = pto.vlds(ub, offset)
```

### Untyped Loop Carried State

```
Error: loop-carried value must have explicit machine type
```

**Solution:** Add type annotations to loop-carried variables:

```python
# Wrong:
remaining = 1024  # Plain Python int
for i in range(0, N, step):
    mask, remaining = pto.make_mask(pto.f32, remaining)

# Correct:
remaining: pto.i32 = 1024
# or
remaining = pto.i32(1024)
```

## Compatibility Notes

The current experimental implementation in `python/pto/dialects/pto.py` differs from this specification in several ways:

1. **Mask types**: The experimental version uses untyped `mask` instead of `mask_b8`/`mask_b16`/`mask_b32`
2. **Barrier operation**: Uses `pto.barrier()` instead of `pto.pipe_barrier()`
3. **MemRef support**: Does not yet support `pto.memref()` types
4. **Operation coverage**: Implements only a subset of operations

When implementing new code, follow this specification. The experimental implementation will be updated to match over time.

## Next Steps

- Explore the ISA documentation in `docs/isa/` for detailed operation semantics
- Check `test/samples/` for example kernels
- Refer to `docs/vpto-spec.md` for the underlying VPTO instruction specification

For compiler developers, see `docs/PTO_IR_manual.md` for MLIR-level details.