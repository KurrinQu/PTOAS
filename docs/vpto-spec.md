# VPTO Spec

Updated: 2026-03-21

## Table Of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Example: Abs](#example-abs)
- [Scope](#scope)
- [ISA Contract](#isa-contract)
- [Core Types](#core-types)
- [Architectural Operand Classes](#architectural-operand-classes)
- [Address Space Conventions](#address-space-conventions)
- [Element Type Constraints](#element-type-constraints)
- [Special Types](#special-types)
- [Implemented String Constraints](#implemented-string-constraints)
- [__VEC_SCOPE__](#vec_scope)
- [Correspondence Categories](#correspondence-categories)
- [1. Sync And Buffer Control](#1-sync-and-buffer-control)
- [2. Copy Programming](#2-copy-programming)
- [3. Copy Transfers](#3-copy-transfers)
- [4. Vector, Predicate And Align Loads](#4-vector-predicate-and-align-loads)
- [5. Materialization And Predicate Construction](#5-materialization-and-predicate-construction)
- [6. Unary Vector Ops](#6-unary-vector-ops)
- [7. Binary Vector Ops](#7-binary-vector-ops)
- [8. Vec-Scalar Ops](#8-vec-scalar-ops)
- [9. Carry, Compare And Select](#9-carry-compare-and-select)
- [10. Pairing And Interleave](#10-pairing-and-interleave)
- [11. Conversion, Index And Sort](#11-conversion-index-and-sort)
- [12. Extended Arithmetic](#12-extended-arithmetic)
- [13. Stateless Stores](#13-stateless-stores)
- [14. Stateful Store Ops](#14-stateful-store-ops)
- [15. Vector Thread And Loop Control](#15-vector-thread-and-loop-control)
- [16. Vector And Predicate Rearrangement](#16-vector-and-predicate-rearrangement)
- [17. Arithmetic Extension Families](#17-arithmetic-extension-families)
- [18. Reduction And Compression Families](#18-reduction-and-compression-families)
- [19. Wide Register Families](#19-wide-register-families)
- [20. Vision And Histogram Families](#20-vision-and-histogram-families)
- [21. Detection And Suppression Families](#21-detection-and-suppression-families)

## Overview

This document defines the Vector PTO (VPTO) Intermediate Representation (IR), a
compiler-internal and externally facing specification designed to represent
vector compute kernels within the PTO architecture. Much like NVVM provides a
robust IR for GPU architectures, VPTO serves as the direct bridge between
high-level programming models and the underlying hardware ISA, providing a
precise, low-level representation of vector workloads explicitly designed for
the Ascend 950 architecture.

### PTO Vector ISA Background

#### Position in the Stack and Layer Modeled

VPTO operates as a very low-level intermediate representation within the PTO
compiler stack. It is uniquely designed to accurately and comprehensively
express all architectural information of the Ascend 950 hardware. It
specifically models the bare-metal vector execution layer, making
hardware-specific capabilities and constraints, such as exact vector lane
configurations, memory space hierarchies, and hardware-specific fusion
semantics, fully transparent and controllable.

#### Why External Developers Read or Author VPTO

While the majority of users will interact with the PTO architecture via
higher-level frameworks, external developers may need to read or author VPTO IR
directly for several key reasons:

- Custom Toolchain Development:
  build custom compiler frontends or domain-specific languages (DSLs) that
  target the Ascend 950 architecture with maximum hardware utilization.
- Performance Engineering:
  inspect the output of high-level compiler passes, verify fine-grained
  optimization behaviors, and pinpoint performance bottlenecks at the
  architectural level.
- Micro-Optimization:
  hand-author highly optimized, critical mathematical kernels using a stable,
  precise IR when higher-level abstractions cannot achieve the theoretical peak
  performance of the hardware.

### Relationship to CCE

VPTO is designed to express the full semantic capabilities of the Compute Cube
Engine (CCE), but with significant structural and pipeline advantages for
compiler development.

- Bypassing the C/Clang Pipeline:
  while CCE heavily relies on C/C++ extensions parsed by Clang, VPTO operates
  entirely independently of the C language frontend. By bypassing Clang AST
  generation and frontend processing, utilizing VPTO significantly reduces
  overall compilation time and memory overhead.
- Enhanced IR Verification:
  because VPTO is a strongly typed, SSA-based (Static Single Assignment)
  compiler IR rather than a C-wrapper API, it provides a much more rigorous and
  detailed IR verification process. Structural inconsistencies, invalid memory
  access patterns, and operand type mismatches are caught immediately with
  precise, explicit diagnostic feedback, providing developers with much higher
  visibility into kernel correctness than traditional CCE error reporting.

### Intended Audience

This document is written for compiler engineers, library writers, and advanced
performance architects. We expect the reader to have a working understanding of
modern compiler infrastructure, specifically MLIR, the principles of Static
Single Assignment (SSA) form, and a deep understanding of the vector-processing
capabilities of the Ascend 950 architecture.

## Getting Started

The Vector PTO (VPTO) IR is architected as a performance-critical layer within the compiler stack, specifically designed to exploit the **Decoupled Access-Execute** (DAE) nature of the Ascend 950 hardware.

### Hardware Pipeline Modeling
The IR is structured to mirror the three primary hardware pipelines of the Ascend 950 architecture. Correct VPTO authoring requires managing the interaction between these asynchronous units:

**MTE2** (Memory Transfer Engine - Inbound): Responsible for moving data from Global Memory (GM) to the Vector tile buffer.

**Vector Core** (Computation): The primary engine for executing SIMD operations on data stored in the Vector tile buffer.

**MTE3** (Memory Transfer Engine - Outbound): Responsible for moving processed data from the Vector tile buffer back to GM.

### Memory and Synchronization Model
VPTO enforces a strict memory hierarchy. The Vector tile buffer is the only valid operand source for vector compute instructions. Consequently, the architecture of a VPTO program is defined by the explicit management of data movement:

**Address Space Isolation**: The IR uses LLVM pointer address spaces to distinguish between GM (`!llvm.ptr<1>`) and the Vector tile buffer (`!llvm.ptr<6>`). The verifier ensures that no compute operation attempts to access GM directly.

**Event-Based Synchronization**: Because the MTE and Vector pipelines operate asynchronously, VPTO utilizes a Flag/Event mechanism. Developers must explicitly insert set_flag and wait_flag operations to resolve Read-After-Write (RAW) and Write-After-Read (WAR) hazards between memory staging and computation.

### Execution Scopes
The IR introduces the concept of the **Vector Function** ({llvm.loop.aivector_scope}). This architectural boundary identifies regions of code where the Vector Core's SIMD capabilities are fully engaged. Inside this scope, the IR provides high-granularity control over vector registers (vreg), predicates (mask), and alignment states (align).

## Example: Abs

Example file:
[build/vpto-doc-abs/Abs/abs-pto.cpp](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/build/vpto-doc-abs/Abs/abs-pto.cpp)

Representative excerpt:

```mlir
pto.vset_loop2_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.vset_loop1_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.vset_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.vcopy_gm_to_ubuf %7, %2, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c0_i64, %c0_i64, %c128_i64, %c128_i64
    {data_select_bit = false, layout = "nd", ub_pad = false}
    : !llvm.ptr<1>, !llvm.ptr<6>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64

pto.vset_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.vwait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

scf.for %dummy = %c0 to %c1 step %c1 {
  scf.for %lane = %c0 to %9 step %c64 {
    %v = pto.vlds %2[%lane] : !llvm.ptr<6> -> !pto.vreg<64xf32>
    %abs = pto.vabs %v : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
    pto.vsts %abs, %8[%lane] : !pto.vreg<64xf32>, !llvm.ptr<6>
  }
} {llvm.loop.aivector_scope}

pto.vset_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.vwait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.vset_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
pto.vset_loop1_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
pto.vset_loop2_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
pto.vcopy_ubuf_to_gm %8, %14, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c128_i64, %c128_i64
    {layout = "nd"}
    : !llvm.ptr<6>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64
```

## Scope

This document is the interface specification for the `mlir::pto` dialect.

It only describes:

- operation names
- operand and result lists
- operand and result types
- important attributes
- corresponding CCE builtin or CCE wrapper family

It does not describe lowering strategy.

## ISA Contract

VPTO is an ISA-level contract without an encoding layer. Each `pto.v*` op below
preserves the full architectural behavior of one ISA instruction or ISA family,
with only the binary encoding removed and with hidden register updates,
predicate flow, and buffer state made explicit in SSA form.

Contract policy:

- `- ISA family:` names the architectural instruction family that defines the
  operation semantics.
- `- semantics:` defines the architecturally visible behavior after removing the
  encoding and implicit register-mutation details.
- `- operand roles:` explains the meaning of each SSA operand, result, and
  control token.
- Chapter-level assertions and exceptions below are normative unless an op
  section narrows them further.

Naming note:

- This spec uses `pto.v*` headings for the exposed VPTO contract.
- Some underlying A5VM op definitions omit that extra `v` prefix for control,
  predicate, and copy helpers; the ISA behavior is unchanged.

## Core Types

- `vreg<T>`: `!pto.vreg<NxT>`
  Fixed-width VPTO vector type with total width exactly 256 bytes.
- `mask`: `!pto.mask`
  Opaque predicate-register type. The element granularity is carried by the producing or consuming opcode family (`*_b8`, `*_b16`, `*_b32`), not by a type parameter on `!pto.mask` itself.
- `align`: `!pto.align`
- `wreg<T>`: `!pto.wreg<LxT>`
  Wide-register payload used by the `W*` instruction families. `L` and `T`
  are family-defined and preserve the widened lane contract spelled by the ISA
  mnemonic suffix, such as `s242u8`, `s482u16`, `s642s32`, or the 48-bit forms
  created by `WCVT48`.
- `buf`: buffer-like LLVM pointer type accepted by the dialect
- `idx`: `index`
- `i32`: `i32`
- `i64`: `i64`

Type parameter conventions used below:

- `!pto.vreg<NxT>`:
  `N` is the lane count, `T` is the element type, and `N * bitwidth(T) = 2048`
- `!pto.wreg<LxT>`:
  `L` is the logical wide-register lane count and `T` is the widened integer
  lane type required by the producing or consuming `W*` family. VPTO MUST keep
  the suffix-defined widening contract; it MUST NOT silently reinterpret a
  `24`, `48`, or `64`-bit wide lane as an ordinary `vreg` lane.
- `!llvm.ptr<AS>`:
  `AS` is the LLVM address space number

## Architectural Operand Classes

The ISA families added later in this document use architectural operand classes
that are broader than the currently materialized `!pto.vreg`, `!pto.mask`, and
`!pto.align` carriers.

- `Xn`, `Xm`, `Xt`, `Xd`:
  main-scalar general registers. In VPTO they carry absolute addresses,
  configuration words, or scalar control values and MUST preserve the integer
  or pointer width required by the consuming family.
- `Sn`, `Sm`, `St`, `Sd`:
  shared-register operands used by vector loops, shift-control forms, scalar
  coefficients, and compact address-generation state. The ISA text constrains
  some forms to 16-bit subwords and others to 32-bit or 64-bit values; VPTO
  MUST model the width required by the named family and MUST preserve even-only
  register-number requirements where the ISA imposes them.
- `Ad` / `VAd`:
  vector address-register state used by address-generation or compatibility
  motion families. VPTO treats this as architectural address state rather than
  as a vector payload.
- `SPR`:
  architecturally named special-register state. Examples in this spec include
  `SQZN`, `AR`, and `RPN_COR_IR`. VPTO MUST preserve SPR-specific update,
  post-update, and clearing semantics exactly; these are not ordinary scalar
  temporaries.
- `PBID`:
  parameter-buffer identifier consumed by vector-thread fetch and release
  control. VPTO treats the identifier as a logical resource handle rather than
  as arbitrary integer data.

Architectural operand classes above describe the ISA contract, not a required
surface syntax for one textual frontend. When a VPTO lowering chooses an SSA
carrier for one of these operands, that carrier MUST preserve the same visible
state transition and validation rules documented here.

## Address Space Conventions

The table below defines the address-space interpretation used by this spec and by the current dialect implementation.

| `AS` | PTO mnemonic | Working interpretation in this spec | Status |
|------|--------------|-------------------------------------|--------|
| `0` | `Zero` | Default / unspecified pointer space; treated as GM-like by the current verifier rules | Normative in this spec |
| `1` | `GM` | Global Memory (GM) | Normative in this spec |
| `2` | `MAT` | Matrix / L1-related storage | Normative in this spec |
| `3` | `LEFT` | Left matrix buffer / L0A-related storage | Normative in this spec |
| `4` | `RIGHT` | Right matrix buffer / L0B-related storage | Normative in this spec |
| `5` | `ACC` | Accumulator / L0C-related storage | Normative in this spec |
| `6` | `VEC` | Vector tile buffer | Normative in this spec |
| `7` | `BIAS` | Bias buffer | Normative in this spec |
| `8` | `SCALING` | Scaling buffer | Normative in this spec |

- Current verifier rule: `!llvm.ptr<0>` and `!llvm.ptr<1>` are treated as GM-like, while `!llvm.ptr<6>` is treated as Vector tile buffer-like.
- External authors should keep the raw numeric LLVM address space in IR and use the symbolic names in this table as the explanatory meaning of those numeric values.

## Element Type Constraints

This section defines how placeholders such as `T`, `T0`, `T1`, and `I` should
be read throughout the spec.

- General vector rule:
  `!pto.vreg<NxT>` requires `T` to be an integer or floating-point element
  type, and `N * bitwidth(T) = 2048`.
- `T`:
  General vector element type accepted by the mapped ISA family. In the current tree this means integer lanes and floating-point lanes such as `i8`, `i16`, `i32`, `i64`, `f16`, `bf16`, and `f32`, subject to the narrower legality of each individual op family.
- `T0`, `T1`:
  Source and result element types for conversion ops. Legal pairs are exactly the pairs implemented by the ISA conversion families `VCVTFI`, `VCVTFF`, `VCVTIF`, `VCVTII`, and `VTRC`; VPTO does not treat `pto.vcvt` as an arbitrary bitcast.
- `I`:
  Integer element type used for offsets, indices, lane selectors, and permutation inputs. Gather, scatter, index-generation, and lane-selection ops require integer vectors; scalar offsets use `index`, `i32`, or `i64` exactly as shown in the op syntax.
- Family-specific exceptions:
  Predicate families use `!pto.mask` rather than `!pto.vreg`; `pto.vmull` returns split widened results; stateful store ops thread `!pto.align` and pointer/index state explicitly; and copy-programming ops are configuration side effects rather than value-producing vector instructions.
  Wide-register families use `!pto.wreg<LxT>` and preserve the suffix-defined
  widening relation between vector-source lanes and wide-register lanes.

## Special Types

### `!pto.mask`

`!pto.mask` models an A5 predicate register, not an integer vector.

Mask data-type expression:

- `!pto.mask` is intentionally unparameterized. Predicate granularity is implied by the op family that creates or consumes it, so `pset_b8`, `pset_b16`, and `pset_b32` all return the same abstract mask type while preserving their ISA-level granularity in the op name.

Use it when an operation needs per-lane enable/disable state.

- producers:
  `pto.vpset_b8`, `pto.vpset_b16`, `pto.vpset_b32`,
  `pto.vpge_b8`, `pto.vpge_b16`, `pto.vpge_b32`,
  `pto.vplds`, `pto.vpld`, `pto.vpldi`,
  `pto.vcmp`, `pto.vcmps`
- consumers:
  `pto.vsel`,
  `pto.vaddc`, `pto.vsubc`, `pto.vaddcs`, `pto.vsubcs`,
  `pto.vpnot`, `pto.vpsel`,
  `pto.vgather2_bc`,
  `pto.vstx2`, `pto.vsstb`,
  `pto.vpsts`, `pto.vpst`, `pto.vpsti`,
  `pto.vpstu`,
  `pto.vmula`

Example:

```mlir
%mask = pto.vcmp %lhs, %rhs, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask
%out = pto.vsel %x, %y, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

### `!pto.align`

`!pto.align` models the A5 vector-align carrier state. It is not payload data.

Use it when an operation needs explicit align-state threading in SSA form.

- producers:
  `pto.vldas`,
  `pto.vpstu`,
  `pto.vstu`,
  `pto.vstus`,
  `pto.vstur`
- consumers:
  `pto.vldus`,
  `pto.vsta`,
  `pto.vstas`,
  `pto.vstar`,
  `pto.vpstu`,
  `pto.vstu`,
  `pto.vstus`,
  `pto.vstur`

Example:

```mlir
%align = pto.vldas %ub[%c0] : !llvm.ptr<6> -> !pto.align
%vec = pto.vldus %align, %ub[%c64] : !pto.align, !llvm.ptr<6> -> !pto.vreg<64xf32>
```

Template placeholder conventions used below:

- `"SRC_PIPE"`, `"DST_PIPE"`:
  string literals such as `"PIPE_MTE2"`, `"PIPE_V"`, `"PIPE_MTE3"`
- `"EVENT_ID"`:
  string literal such as `"EVENT_ID0"`
- `"LAYOUT"`:
  string literal layout selector, for example `"nd"`
- `"DIST"`:
  string literal distribution selector carried by the op
- `"POSITION"`:
  string literal lane-position selector used by `vdup`
- `"MODE"`:
  string literal mode selector used by stateful store / multiply-accumulate ops
- `"ROUND_MODE"`:
  string literal rounding-mode selector
- `"SAT_MODE"`:
  string literal saturation selector
- `"PART_MODE"`:
  string literal half/part selector
- `"ORDER"`:
  string literal order selector used by `vci`
- `"CMP_MODE"`:
  string literal compare predicate selector
- `"PAT_*"`:
  predicate pattern literal accepted by the corresponding predicate op
- `T|!pto.vreg<NxT>`:
  either a scalar `T` or a vector operand `!pto.vreg<NxT>`, matching the op verifier

## Implemented String Constraints

This section records string-valued operands and attributes that are already
checked by the current verifier implementation.

If a token is not listed here, the current dialect usually only requires a
non-empty string or leaves the token unconstrained for now.

### Predicate Patterns

Used by:
`pto.vpset_b8`, `pto.vpset_b16`, `pto.vpset_b32`,
`pto.vpge_b8`, `pto.vpge_b16`, `pto.vpge_b32`

- allowed values:
  `PAT_ALL | PAT_VL1 | PAT_VL2 | PAT_VL3 | PAT_VL4 | PAT_VL8 | PAT_VL16 | PAT_VL32 | PAT_VL64 | PAT_VL128 | PAT_M3 | PAT_M4 | PAT_H | PAT_Q | PAT_ALLF`

### Distribution Tokens

Used by `pto.vlds`:

- allowed values:
  `NORM | BLK | DINTLV_B32 | UNPK_B16`

Used by `pto.vpld`, `pto.vpldi`:

- allowed values:
  `NORM | US | DS`

Used by `pto.vpst`, `pto.vpsti`:

- allowed values:
  `NORM | PK`

Used by `pto.vldx2`:

- allowed values:
  `DINTLV_B8 | DINTLV_B16 | DINTLV_B32 | BDINTLV`

Used by `pto.vstx2`:

- allowed values:
  `INTLV_B8 | INTLV_B16 | INTLV_B32`

### Stride Tokens

Used by `pto.vsld`, `pto.vsst`:

- allowed values:
  `STRIDE_S3_B16 | STRIDE_S4_B64 | STRIDE_S8_B32 | STRIDE_S2_B64 | STRIDE_VSST_S8_B16`

### Compare Modes

Used by `pto.vcmp`, `pto.vcmps`:

- allowed values:
  `eq | ne | lt | le | gt | ge`

### Part Tokens

Used by `pto.vintlvv2`, `pto.vdintlvv2`:

- allowed values:
  `LOWER | HIGHER`

Current restricted subset:

- `pto.vppack`: only `LOWER`
- `pto.vpunpack`: only `LOWER`

### Mode Tokens

Used by `pto.vmula`:

- allowed values:
  `MODE_ZEROING | MODE_UNKNOWN | MODE_MERGING`

Used by `pto.vstu`, `pto.vstus`, `pto.vstur`:

- allowed values:
  `POST_UPDATE | NO_POST_UPDATE`

### Conversion Control Tokens

Used by `pto.vcvt.round_mode`:

- allowed values:
  `ROUND_R | ROUND_A | ROUND_F | ROUND_C | ROUND_Z | ROUND_O`

Used by `pto.vcvt.sat`:

- allowed values:
  `RS_ENABLE | RS_DISABLE`

Used by `pto.vcvt.part`:

- allowed values:
  `PART_EVEN | PART_ODD`

### Not Yet Enumerated In Verifier

The following placeholders appear in syntax templates but are not yet fully
enumerated by the verifier:

- `"LAYOUT"`
- `"POSITION"`
- `"ORDER"`
- `"SRC_PIPE"`, `"DST_PIPE"`, `"EVENT_ID"`

### `LAYOUT`

- Current repo-defined layout spellings are `nd`, `dn`, and `nz`.
- Copy ops preserve the layout token as part of the transfer contract.
- The verifier does not yet exhaustively cross-check every layout-sensitive combination, so producers must only emit layout values that the selected copy helper or backend path actually implements.

### `POSITION`

- `POSITION` selects which `VDUP*` source position is duplicated when the input is a vector.
- The current verifier checks type compatibility but does not enumerate a closed token set, so this field is an implementation-defined token that must be preserved exactly by translators.

### `ORDER`

- `ORDER` selects the lane-index generation order for `VCI`.
- The currently documented token is `INC_ORDER`, which produces monotonic increasing lane indices.
- The current verifier does not enforce a closed enum for this field, so any alternative order token must be introduced together with matching lowering support.

### `SRC_PIPE` / `DST_PIPE`

- Legal pipe names in the current tree are `PIPE_S`, `PIPE_V`, `PIPE_M`, `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_ALL`, `PIPE_MTE4`, `PIPE_MTE5`, `PIPE_V2`, `PIPE_FIX`, `VIRTUAL_PIPE_MTE2_L1A`, and `VIRTUAL_PIPE_MTE2_L1B`.
- `SRC_PIPE` names the producer side of the dependency and `DST_PIPE` names the consumer side.
- A `wait_flag` must use the same source pipe, destination pipe, and event id triplet that the corresponding `set_flag` published.

### `EVENT_ID`

- Legal event identifiers in the current tree are `EVENT_ID0` through `EVENT_ID7`.
- The event id is not meaningful by itself; it is interpreted together with the `(SRC_PIPE, DST_PIPE)` pair.
- Producer and consumer sides must agree on the entire triplet `(SRC_PIPE, DST_PIPE, EVENT_ID)` for synchronization to be well formed.

## Architectural Assertions And Exceptions

The following rules are ISA-level requirements that VPTO preserves even though
it does not encode the original instruction words.

Architectural assertions:

- Vector compute, gather, scatter, predicate-load, and predicate-store families operate on Vector tile buffer-backed storage unless an op section states otherwise.
- Distribution tokens, stride tokens, part selectors, and predicate patterns are semantically significant; changing them changes the operation, not just the encoding.
- Alignment requirements are part of the contract. Violating the distribution-specific address-alignment rule raises an exception.
- Stateful align-register behavior is explicit in VPTO through `!pto.align` values and state-threading results, not by implicit mutation of a hidden architectural register.
- Copy helper families preserve layout, burst geometry, and stride semantics as architecturally visible behavior.

Architectural exceptions:

- Vector tile buffer access overflow raises an exception for reads or writes that exceed the permitted Vector tile buffer address range.
- Load and store families raise an exception when ISA-required alignment constraints are violated.
- Memory-to-memory sorter and filter families require their ISA alignment and non-overlap constraints; violating those constraints raises an exception.
- `INF` and `NaN` in source operands raise an exception for the arithmetic, conversion, and sort families in this spec wherever the ISA defines those inputs as exceptional.
- Division by `+0` or `-0` raises an exception for `pto.vdiv` and `pto.vrec`; the same ISA rule also applies to reciprocal-square-root families not currently exposed here.
- Negative input raises an exception for `pto.vln` and `pto.vsqrt`; the same ISA rule also applies to reciprocal-square-root families not currently exposed here.
- Certain conversions from negative source values to unsigned integer destinations, including forms such as `f16 -> u8` and `s32 -> u16/u8`, raise an exception rather than silently wrapping.

## __VEC_SCOPE__

`__VEC_SCOPE__` is not an `pto` op.

It must be represented as:

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
scf.for %dummy = %c0 to %c1 step %c1 {
  // vector-scope body
} {llvm.loop.aivector_scope}
```

This is the dialect-level representation of the A5 vector-scope loop.

## Correspondence Categories

- `direct builtin`
  The op maps naturally to one CCE builtin family, usually `__builtin_cce_<name>_*`.
- `wrapper family`
  The op corresponds to a CCE wrapper family, but the wrapper may dispatch to
  multiple builtin spellings depending on type, architecture, or mode.

Builtin naming policy in this document:

- if a visible CCE intrinsic is declared as
  `clang_builtin_alias(__builtin_cce_...)`, the spec lists the builtin name
  explicitly
- if PTO A5 code calls a wrapper function that internally composes several
  intrinsics or builtins, the spec lists both the wrapper name and the visible
  builtin family

## 1. Sync And Buffer Control

### `pto.vset_flag`

- syntax:
  `pto.vset_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- operand roles:
  `"SRC_PIPE"` names the producer pipeline, `"DST_PIPE"` names the consumer pipeline, and `"EVENT_ID"` names the event channel being published.
- ISA family:
  `SET_FLAG`
- semantics:
  Publishes `EVENT_ID` from `SRC_PIPE` to `DST_PIPE` so later waits can order the asynchronous pipelines.
- CCE correspondence:
  `set_flag(pipe_t, pipe_t, event_t|uint64_t)`
  `__builtin_cce_set_flag`
  PTO token path:
  `__pto_set_flag`
  `__builtin_cce_tile_set_flag`

### `pto.vwait_flag`

- syntax:
  `pto.vwait_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- operand roles:
  `"SRC_PIPE"` names the producer pipeline, `"DST_PIPE"` names the consumer pipeline, and `"EVENT_ID"` names the event channel being waited on.
- ISA family:
  `WAIT_FLAG`
- semantics:
  Stalls the consumer side until the matching `(SRC_PIPE, DST_PIPE, EVENT_ID)` event has been observed.
- CCE correspondence:
  `wait_flag(pipe_t, pipe_t, event_t|uint64_t)`
  `__builtin_cce_wait_flag`
  PTO token path:
  `__pto_wait_flag`
  `__builtin_cce_tile_wait_flag`

### `pto.vpipe_barrier`

- syntax:
  `pto.vpipe_barrier "PIPE_*"`
- operand roles:
  `"PIPE_*"` selects the pipeline whose in-order execution is being fenced.
- ISA family:
  `PIPE_BARRIER`
- semantics:
  Inserts a same-pipe execution barrier; later operations on `PIPE_*` cannot overtake earlier ones on that pipeline.
- CCE correspondence:
  `pipe_barrier(pipe_t)`
  `__builtin_cce_pipe_barrier`

### `pto.vget_buf`

- syntax:
  `pto.vget_buf "PIPE_*", %buf_id, %mode : i64, i64`
- operand roles:
  `"PIPE_*"` selects the owning pipeline, `%buf_id` is the buffer identifier being requested, and `%mode` carries the hardware acquisition mode.
- ISA family:
  `GET_BUF`
- semantics:
  Acquires the hardware buffer token on the selected pipe and reserves the identified buffer slot or mode-controlled token for later use.
- CCE correspondence:
  `get_buf(pipe_t, uint8_t|uint64_t, bool)`
  `__builtin_cce_get_buf`

### `pto.vrls_buf`

- syntax:
  `pto.vrls_buf "PIPE_*", %buf_id, %mode : i64, i64`
- operand roles:
  `"PIPE_*"` selects the owning pipeline, `%buf_id` is the buffer identifier being released, and `%mode` carries the hardware release mode.
- ISA family:
  `RLS_BUF`
- semantics:
  Releases the hardware buffer token on the selected pipe and returns the previously acquired buffer slot or mode-controlled token to the implementation.
- CCE correspondence:
  `rls_buf(pipe_t, uint8_t|uint64_t, bool)`
  `__builtin_cce_rls_buf`

## 2. Copy Programming

### `pto.vset_loop2_stride_outtoub`

- syntax:
  `pto.vset_loop2_stride_outtoub %first, %second : i64, i64`
- operand roles:
  `%first` and `%second` are the two i64 configuration fields consumed by the corresponding copy-programming register pair.
- ISA family:
  `SET_LOOP2_STRIDE_OUTTOUB`
- semantics:
  Sets the outer-loop stride consumed by the next GM-to-Vector-tile-buffer transfer sequence.
- CCE correspondence:
  `set_loop2_stride_outtoub(uint64_t)`
  `__builtin_cce_set_loop2_stride_outtoub`

### `pto.vset_loop1_stride_outtoub`

- syntax:
  `pto.vset_loop1_stride_outtoub %first, %second : i64, i64`
- operand roles:
  `%first` and `%second` are the two i64 configuration fields consumed by the corresponding copy-programming register pair.
- ISA family:
  `SET_LOOP1_STRIDE_OUTTOUB`
- semantics:
  Sets the inner-loop stride consumed by the next GM-to-Vector-tile-buffer transfer sequence.
- CCE correspondence:
  `set_loop1_stride_outtoub(uint64_t)`
  `__builtin_cce_set_loop1_stride_outtoub`

### `pto.vset_loop_size_outtoub`

- syntax:
  `pto.vset_loop_size_outtoub %first, %second : i64, i64`
- operand roles:
  `%first` and `%second` are the two i64 configuration fields consumed by the corresponding copy-programming register pair.
- ISA family:
  `SET_LOOP_SIZE_OUTTOUB`
- semantics:
  Sets the loop extents consumed by the next GM-to-Vector-tile-buffer transfer sequence.
- CCE correspondence:
  `set_loop_size_outtoub(uint64_t)`
  `__builtin_cce_set_loop_size_outtoub`

### `pto.vset_loop2_stride_ubtoout`

- syntax:
  `pto.vset_loop2_stride_ubtoout %first, %second : i64, i64`
- operand roles:
  `%first` and `%second` are the two i64 configuration fields consumed by the corresponding copy-programming register pair.
- ISA family:
  `SET_LOOP2_STRIDE_UBTOOUT`
- semantics:
  Sets the outer-loop stride consumed by the next Vector-tile-buffer-to-GM transfer sequence.
- CCE correspondence:
  `set_loop2_stride_ubtoout(uint64_t)`
  `__builtin_cce_set_loop2_stride_ubtoout`

### `pto.vset_loop1_stride_ubtoout`

- syntax:
  `pto.vset_loop1_stride_ubtoout %first, %second : i64, i64`
- operand roles:
  `%first` and `%second` are the two i64 configuration fields consumed by the corresponding copy-programming register pair.
- ISA family:
  `SET_LOOP1_STRIDE_UBTOOUT`
- semantics:
  Sets the inner-loop stride consumed by the next Vector-tile-buffer-to-GM transfer sequence.
- CCE correspondence:
  `set_loop1_stride_ubtoout(uint64_t)`
  `__builtin_cce_set_loop1_stride_ubtoout`

### `pto.vset_loop_size_ubtoout`

- syntax:
  `pto.vset_loop_size_ubtoout %first, %second : i64, i64`
- operand roles:
  `%first` and `%second` are the two i64 configuration fields consumed by the corresponding copy-programming register pair.
- ISA family:
  `SET_LOOP_SIZE_UBTOOUT`
- semantics:
  Sets the loop extents consumed by the next Vector-tile-buffer-to-GM transfer sequence.
- CCE correspondence:
  `set_loop_size_ubtoout(uint64_t)`
  `__builtin_cce_set_loop_size_ubtoout`

## 3. Copy Transfers

### `pto.vcopy_gm_to_ubuf`

- syntax:
  `pto.vcopy_gm_to_ubuf %source, %destination, %valid_rows, %valid_cols, %sid, %n_burst, %len_burst, %left_padding_count, %right_padding_count, %l2_cache_ctl, %gm_stride, %ub_stride {layout = "LAYOUT", data_select_bit = true|false, ub_pad = true|false} : !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64`
- operand roles:
  `%source` is the GM base pointer, `%destination` is the Vector tile buffer base pointer, `%valid_rows` and `%valid_cols` describe the logical tile extent, `%sid` is the stream or source identifier, `%n_burst` and `%len_burst` describe the burst geometry, `%left_padding_count` and `%right_padding_count` describe edge padding, `%l2_cache_ctl` carries cache control bits, `%gm_stride` and `%ub_stride` are per-burst strides, and `"LAYOUT"` records the transfer layout token.
- ISA family:
  `GM->Vector-tile-buffer copy helper family`
- semantics:
  Copies a 2-D burst tile with optional padding and layout metadata into the Vector tile buffer.
- CCE correspondence:
  `copy_gm_to_ubuf(...)`
  PTO A5 path commonly uses `copy_gm_to_ubuf_align_v2(...)`
  `__builtin_cce_copy_gm_to_ubuf_align_v2`
  composed loop intrinsics:
  `__builtin_cce_set_loop2_stride_outtoub`
  `__builtin_cce_set_loop1_stride_outtoub`
  `__builtin_cce_set_loop_size_outtoub`

### `pto.vcopy_ubuf_to_ubuf`

- syntax:
  `pto.vcopy_ubuf_to_ubuf %source, %destination, %sid, %n_burst, %len_burst, %src_stride, %dst_stride : !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64, i64, i64, i64`
- operand roles:
  `%source` and `%destination` are Vector tile buffer base pointers, `%sid` is the stream identifier, `%n_burst` and `%len_burst` describe the burst geometry, and `%src_stride` and `%dst_stride` are the per-burst source and destination strides.
- ISA family:
  `Vector-tile-buffer->Vector-tile-buffer copy helper family`
- semantics:
  Copies data between two Vector tile buffer-backed buffers using the stated burst and stride parameters.
- CCE correspondence:
  `copy_ubuf_to_ubuf(...)`
  `__builtin_cce_copy_ubuf_to_ubuf`

### `pto.vcopy_ubuf_to_gm`

- syntax:
  `pto.vcopy_ubuf_to_gm %source, %destination, %valid_rows, %valid_cols, %sid, %n_burst, %len_burst, %reserved, %burst_dst_stride, %burst_src_stride {layout = "LAYOUT"} : !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64, i64, i64, i64, i64, i64, i64`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%destination` is the GM base pointer, `%valid_rows` and `%valid_cols` describe the logical tile extent, `%sid` is the stream identifier, `%n_burst` and `%len_burst` describe the burst geometry, `%reserved` is the ISA-reserved field carried by the helper path, `%burst_dst_stride` and `%burst_src_stride` are the per-burst strides, and `"LAYOUT"` records the transfer layout token.
- ISA family:
  `Vector-tile-buffer->GM copy helper family`
- semantics:
  Writes a 2-D burst tile from the Vector tile buffer back to GM.
- CCE correspondence:
  `copy_ubuf_to_gm(...)`
  PTO A5 path commonly uses `copy_ubuf_to_gm_align_v2(...)`
  `__builtin_cce_copy_ubuf_to_gm_align_v2`
  composed loop intrinsics:
  `__builtin_cce_set_loop2_stride_ubtoout`
  `__builtin_cce_set_loop1_stride_ubtoout`
  `__builtin_cce_set_loop_size_ubtoout`

## 4. Vector, Predicate And Align Loads

ISA assertions for this family:

- These ops read from Vector tile buffer-backed storage; GM-backed sources are not valid vector-load operands in VPTO.
- Alignment requirements are part of the ISA contract and depend on the selected distribution token.
- `pto.vldas` initializes align state for subsequent unaligned accesses; `pto.vldus` assumes a valid align chain for the same logical stream.
- ISA forms that post-update a shared register are represented in VPTO by explicit SSA operands or by state-threading ops rather than by hidden base-pointer mutation.


### `pto.vlds`

- syntax:
  `%result = pto.vlds %source[%offset] {dist = "DIST"} : !llvm.ptr<AS> -> !pto.vreg<NxT>`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offset` is the load displacement from that base, `"DIST"` selects the ISA distribution mode, and `%result` is the loaded vector.
- ISA family:
  `VLD` / `VLDS`
- semantics:
  Loads one vector from the Vector tile buffer at `source + offset` using the aligned-load form selected by `DIST`. The distribution token determines the lane arrangement and alignment requirement of the access, and any ISA-defined base update is represented outside this op rather than as an implicit side effect.
- CCE correspondence:
  `vld(...)`, `vlds(...)`
  `__builtin_cce_vldsx1_*`
  related extended families:
  `__builtin_cce_vldix1_*`, `__builtin_cce_vldsx1_post_*`

### `pto.vldas`

- syntax:
  `%result = pto.vldas %source[%offset] : !llvm.ptr<AS> -> !pto.align`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offset` is the align-initialization displacement, and `%result` is the produced align state.
- ISA family:
  `VLDAS`
- semantics:
  Initializes align state for a subsequent unaligned load stream. The seeded state is derived from the aligned base corresponding to `source + offset`, with the ISA-required low address bits removed.
- CCE correspondence:
  `vldas(...)`
  `__builtin_cce_vldas_*`

### `pto.vldus`

- syntax:
  `%result = pto.vldus %align, %source[%offset] : !pto.align, !llvm.ptr<AS> -> !pto.vreg<NxT>`
- operand roles:
  `%align` is the incoming align state, `%source` is the Vector tile buffer base pointer, `%offset` is the load displacement, and `%result` is the assembled vector result.
- ISA family:
  `VLDUS`
- semantics:
  Loads one unaligned vector by combining `%align` with the aligned data fetched from `source + offset`. `%align` must come from the same logical align stream, and the returned vector is the ISA contiguous lane sequence assembled across the alignment boundary.
- CCE correspondence:
  `vldus(...)`
  `__builtin_cce_vldus_*`, `__builtin_cce_vldus_post_*`

### `pto.vplds`

- syntax:
  `%result = pto.vplds %source[%offset] {dist = "DIST"} : !llvm.ptr<AS> -> !pto.mask`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offset` is the load displacement, `"DIST"` selects the predicate-load distribution, and `%result` is the loaded predicate.
- ISA family:
  `PLDS`
- semantics:
  Loads predicate state from the Vector tile buffer at `source + offset` using the selected predicate-load distribution.
- CCE correspondence:
  `plds(...)`
  `__builtin_cce_plds_b8`

### `pto.vpld`

- syntax:
  `%result = pto.vpld %source[%offset], "DIST" : !llvm.ptr<AS>, index -> !pto.mask`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offset` is the index-style displacement, `"DIST"` selects the predicate-load distribution, and `%result` is the loaded predicate.
- ISA family:
  `PLD`
- semantics:
  Loads predicate state from the Vector tile buffer using an explicit index offset and predicate-load distribution token.
- CCE correspondence:
  `pld(...)`
  `__builtin_cce_pld_b8`

### `pto.vpldi`

- syntax:
  `%result = pto.vpldi %source, %offset, "DIST" : !llvm.ptr<AS>, i32 -> !pto.mask`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offset` is the immediate-style scalar displacement, `"DIST"` selects the predicate-load distribution, and `%result` is the loaded predicate.
- ISA family:
  `PLDI`
- semantics:
  Loads predicate state from the Vector tile buffer using an immediate-style scalar offset and predicate-load distribution token.
- CCE correspondence:
  `pldi(...)`
  `__builtin_cce_pldi_b8`, `__builtin_cce_pldi_post_b8`

### `pto.vldx2`

- syntax:
  `%low, %high = pto.vldx2 %source[%offset], "DIST" : !llvm.ptr<AS>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offset` is the displacement, `"DIST"` selects the x2 load distribution token, and `%low` and `%high` are the two produced vector results.
- ISA family:
  `Dual-result aligned-load distributions for VLD variants`
- semantics:
  Loads one Vector tile buffer vector stream at `source + offset` and splits the result into `%low` and `%high` according to the x2 distribution selected by `DIST`. The distribution token defines how lanes are deinterleaved between the two returned vectors.
- CCE correspondence:
  `vld(...)`
  `__builtin_cce_vldx2_*`

### `pto.vgather2`

- syntax:
  `%result = pto.vgather2 %source, %offsets, %active_lanes : !llvm.ptr<AS>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offsets` is the per-lane offset vector, `%active_lanes` bounds how many lanes participate, and `%result` is the gathered vector.
- ISA family:
  `VGATHER2`
- semantics:
  For each active lane `i` with `i < active_lanes`, loads the element at Vector tile buffer address `source + offsets[i]` and writes it to result lane `i`.
- CCE correspondence:
  `vgather2(...)`
  `__builtin_cce_vgather2_*`, `__builtin_cce_vgather2_v300_*`

### `pto.vgatherb`

- syntax:
  `%result = pto.vgatherb %source, %offsets, %active_lanes : !llvm.ptr<AS>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offsets` is the per-lane offset vector, `%active_lanes` bounds how many lanes participate, and `%result` is the gathered vector.
- ISA family:
  `VGATHERB`
- semantics:
  For each active lane `i` with `i < active_lanes`, loads the byte-granular element at Vector tile buffer address `source + offsets[i]` and writes it to result lane `i`.
- CCE correspondence:
  `vgatherb(...)`
  `__builtin_cce_vgatherb_*`, `__builtin_cce_vgatherb_v300_*`, `__builtin_cce_vgatherb_v310_*`

### `pto.vgather2_bc`

- syntax:
  `%result = pto.vgather2_bc %source, %offsets, %mask : !llvm.ptr<AS>, !pto.vreg<NxI>, !pto.mask -> !pto.vreg<NxT>`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offsets` is the per-lane offset vector, `%mask` selects which lanes participate, and `%result` is the gathered vector.
- ISA family:
  `VGATHER2_BC`
- semantics:
  For each lane enabled by `%mask`, loads the element at Vector tile buffer address `source + offsets[i]` and writes it to result lane `i`.
- CCE correspondence:
  `vgather2_bc(...)`
  `__builtin_cce_vgather2_bc_*`

### `pto.vsld`

- syntax:
  `%result = pto.vsld %source[%offset], "STRIDE" : !llvm.ptr<AS> -> !pto.vreg<NxT>`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offset` is the displacement, `"STRIDE"` selects the strided-load token, and `%result` is the loaded vector.
- ISA family:
  `VSLD`
- semantics:
  Loads `%result` from the Vector tile buffer using the address progression encoded by `STRIDE`. The stride token determines the spacing between consecutive source elements or packed groups.
- CCE correspondence:
  `vsld(...)`
  `__builtin_cce_vsld_*`

### `pto.vsldb`

- syntax:
  `%result = pto.vsldb %source, %offset, %mask : !llvm.ptr<AS>, i32, !pto.mask -> !pto.vreg<NxT>`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offset` is the scalar displacement, `%mask` is the predicate control, and `%result` is the loaded vector.
- ISA family:
  `VSLDB`
- semantics:
  Loads a strided vector from the Vector tile buffer using the scalar displacement `%offset` and writes only the lanes enabled by `%mask`.
- CCE correspondence:
  `vsldb(...)`
  `__builtin_cce_vsldb_*`, `__builtin_cce_vsldb_post_*`

## 5. Materialization And Predicate Construction

### `pto.vbr`

- syntax:
  `%result = pto.vbr %value : T -> !pto.vreg<NxT>`
- operand roles:
  `%value` is the scalar value broadcast into all lanes and `%result` is the produced vector.
- ISA family:
  `VBR`
- semantics:
  Broadcasts one scalar value across all lanes of the result vector.
- CCE correspondence:
  broadcast/materialization family used by PTO scalar-to-vector expansion

### `pto.vdup`

- syntax:
  `%result = pto.vdup %input {position = "POSITION", mode = "MODE"} : T|!pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is either the scalar source or the source vector, `"POSITION"` selects the lane position when duplicating from a vector, `"MODE"` carries the duplication mode token, and `%result` is the duplicated vector.
- ISA family:
  `VDUP` / `VDUPS` / `VDUPI` / `VDUPM`
- semantics:
  Duplicates a scalar input or the lane selected by `POSITION` into every lane of `%result`, according to the duplication form selected by `MODE`.
- CCE correspondence:
  `vdup(...)`
  `__builtin_cce_vdup_*`

### `pto.vpset_b8`

- syntax:
  `%result = pto.vpset_b8 "PAT_*" : !pto.mask`
- operand roles:
  `"PAT_*"` selects the ISA predicate pattern and `%result` is the produced predicate register.
- ISA family:
  `PSET`
- semantics:
  Creates a predicate register in 8-bit granularity from the selected `PAT_*` pattern.
- CCE correspondence:
  `pset_b8(...)`
  `__builtin_cce_pset_b8`

### `pto.vpset_b16`

- syntax:
  `%result = pto.vpset_b16 "PAT_*" : !pto.mask`
- operand roles:
  `"PAT_*"` selects the ISA predicate pattern and `%result` is the produced predicate register.
- ISA family:
  `PSET`
- semantics:
  Creates a predicate register in 16-bit granularity from the selected `PAT_*` pattern.
- CCE correspondence:
  `pset_b16(...)`
  `__builtin_cce_pset_b16`

### `pto.vpset_b32`

- syntax:
  `%result = pto.vpset_b32 "PAT_*" : !pto.mask`
- operand roles:
  `"PAT_*"` selects the ISA predicate pattern and `%result` is the produced predicate register.
- ISA family:
  `PSET`
- semantics:
  Creates a predicate register in 32-bit granularity from the selected `PAT_*` pattern.
- CCE correspondence:
  `pset_b32(...)`
  `__builtin_cce_pset_b32`

### `pto.vpge_b8`

- syntax:
  `%result = pto.vpge_b8 "PAT_*" : !pto.mask`
- operand roles:
  `"PAT_*"` selects the ISA predicate pattern and `%result` is the produced predicate register.
- ISA family:
  `PGE`
- semantics:
  Creates an 8-bit-granularity prefix predicate from the selected `PAT_*` pattern.
- CCE correspondence:
  `pge_b8(...)`
  `__builtin_cce_pge_b8`

### `pto.vpge_b16`

- syntax:
  `%result = pto.vpge_b16 "PAT_*" : !pto.mask`
- operand roles:
  `"PAT_*"` selects the ISA predicate pattern and `%result` is the produced predicate register.
- ISA family:
  `PGE`
- semantics:
  Creates a 16-bit-granularity prefix predicate from the selected `PAT_*` pattern.
- CCE correspondence:
  `pge_b16(...)`
  `__builtin_cce_pge_b16`

### `pto.vpge_b32`

- syntax:
  `%result = pto.vpge_b32 "PAT_*" : !pto.mask`
- operand roles:
  `"PAT_*"` selects the ISA predicate pattern and `%result` is the produced predicate register.
- ISA family:
  `PGE`
- semantics:
  Creates a 32-bit-granularity prefix predicate from the selected `PAT_*` pattern.
- CCE correspondence:
  `pge_b32(...)`
  `__builtin_cce_pge_b32`

### `pto.vppack`

- syntax:
  `%result = pto.vppack %input, "PART" : !pto.mask -> !pto.mask`
- operand roles:
  `%input` is the source predicate, `"PART"` selects which packed half is addressed, and `%result` is the transformed predicate.
- ISA family:
  `PPACK`
- semantics:
  Compresses the predicate lanes selected by `PART` into the packed predicate representation returned in `%result`.
- CCE correspondence:
  `ppack(...)`

### `pto.vpunpack`

- syntax:
  `%result = pto.vpunpack %input, "PART" : !pto.mask -> !pto.mask`
- operand roles:
  `%input` is the source predicate, `"PART"` selects which packed half is addressed, and `%result` is the transformed predicate.
- ISA family:
  `PUNPACK`
- semantics:
  Expands the packed predicate lanes selected by `PART` into the unpacked predicate representation returned in `%result`.
- CCE correspondence:
  `punpack(...)`

## 6. Unary Vector Ops

### `pto.vabs`

- syntax:
  `%result = pto.vabs %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VABS`
- semantics:
  Applies lane-wise absolute value to the input vector.
- CCE correspondence:
  `vabs(...)`
  `__builtin_cce_vabs_*`

### `pto.vexp`

- syntax:
  `%result = pto.vexp %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VEXP`
- semantics:
  Applies lane-wise exponential to the input vector.
- CCE correspondence:
  `vexp(...)`
  `__builtin_cce_vexp_*`

### `pto.vln`

- syntax:
  `%result = pto.vln %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VLN`
- semantics:
  Applies lane-wise natural logarithm to the input vector.
- CCE correspondence:
  `vln(...)`
  `__builtin_cce_vln_*`

### `pto.vsqrt`

- syntax:
  `%result = pto.vsqrt %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VSQRT`
- semantics:
  Applies lane-wise square root to the input vector.
- CCE correspondence:
  `vsqrt(...)`
  `__builtin_cce_vsqrt_*`

### `pto.vrec`

- syntax:
  `%result = pto.vrec %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VREC`
- semantics:
  Applies lane-wise reciprocal to the input vector.
- CCE correspondence:
  `vrec(...)`
  `__builtin_cce_vrec_*`

### `pto.vrelu`

- syntax:
  `%result = pto.vrelu %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VRELU`
- semantics:
  Applies lane-wise rectified-linear activation to the input vector.
- CCE correspondence:
  `vrelu(...)`
  `__builtin_cce_vrelu_*`

### `pto.vnot`

- syntax:
  `%result = pto.vnot %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VNOT`
- semantics:
  Applies lane-wise bitwise logical inversion to the input vector.
- CCE correspondence:
  `vnot(...)`
  `__builtin_cce_vnot_*`

### `pto.vcadd`

- syntax:
  `%result = pto.vcadd %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VCADD`
- semantics:
  Performs reduction-add within each ISA reduction group and returns the vector-shaped partial results.
- CCE correspondence:
  `vcadd(...)`
  `__builtin_cce_vcadd_*`

### `pto.vcmax`

- syntax:
  `%result = pto.vcmax %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VCMAX`
- semantics:
  Performs reduction-max within each ISA reduction group and returns the vector-shaped partial results.
- CCE correspondence:
  `vcmax(...)`
  `__builtin_cce_vcmax_*`

### `pto.vcmin`

- syntax:
  `%result = pto.vcmin %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VCMIN`
- semantics:
  Performs reduction-min within each ISA reduction group and returns the vector-shaped partial results.
- CCE correspondence:
  `vcmin(...)`
  `__builtin_cce_vcmin_*`

### `pto.vbcnt`

- syntax:
  `%result = pto.vbcnt %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VBCNT`
- semantics:
  Computes the lane-wise bit population count.
- CCE correspondence:
  `vbcnt(...)`
  `__builtin_cce_vbcnt_*`

### `pto.vcls`

- syntax:
  `%result = pto.vcls %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VCLS`
- semantics:
  Computes the lane-wise count of leading sign bits.
- CCE correspondence:
  `vcls(...)`
  `__builtin_cce_vcls_*`

## 7. Binary Vector Ops

### `pto.vadd`

- syntax:
  `%result = pto.vadd %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VADD`
- semantics:
  Computes lane-wise addition of `%lhs` and `%rhs`.
- CCE correspondence:
  `vadd(...)`
  `__builtin_cce_vadd_*`

### `pto.vsub`

- syntax:
  `%result = pto.vsub %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VSUB`
- semantics:
  Computes lane-wise subtraction of `%rhs` from `%lhs`.
- CCE correspondence:
  `vsub(...)`
  `__builtin_cce_vsub_*`

### `pto.vmul`

- syntax:
  `%result = pto.vmul %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VMUL`
- semantics:
  Computes lane-wise multiplication of `%lhs` and `%rhs`.
- CCE correspondence:
  `vmul(...)`
  `__builtin_cce_vmul_*`

### `pto.vdiv`

- syntax:
  `%result = pto.vdiv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VDIV`
- semantics:
  Computes lane-wise division of `%lhs` by `%rhs`.
- CCE correspondence:
  `vdiv(...)`
  `__builtin_cce_vdiv_*`

### `pto.vmax`

- syntax:
  `%result = pto.vmax %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VMAX`
- semantics:
  Computes the lane-wise maximum of `%lhs` and `%rhs`.
- CCE correspondence:
  `vmax(...)`
  `__builtin_cce_vmax_*`

### `pto.vmin`

- syntax:
  `%result = pto.vmin %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VMIN`
- semantics:
  Computes the lane-wise minimum of `%lhs` and `%rhs`.
- CCE correspondence:
  `vmin(...)`
  `__builtin_cce_vmin_*`

### `pto.vand`

- syntax:
  `%result = pto.vand %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VAND`
- semantics:
  Computes lane-wise bitwise AND of `%lhs` and `%rhs`.
- CCE correspondence:
  `vand(...)`
  `__builtin_cce_vand_*`

### `pto.vor`

- syntax:
  `%result = pto.vor %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VOR`
- semantics:
  Computes lane-wise bitwise OR of `%lhs` and `%rhs`.
- CCE correspondence:
  `vor(...)`
  `__builtin_cce_vor_*`

### `pto.vxor`

- syntax:
  `%result = pto.vxor %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VXOR`
- semantics:
  Computes lane-wise bitwise XOR of `%lhs` and `%rhs`.
- CCE correspondence:
  `vxor(...)`
  `__builtin_cce_vxor_*`

### `pto.vshl`

- syntax:
  `%result = pto.vshl %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VSHL`
- semantics:
  Shifts each lane of `%lhs` left by the amount carried in the corresponding `%rhs` lane.
- CCE correspondence:
  `vshl(...)`
  `__builtin_cce_vshl_*`

### `pto.vshr`

- syntax:
  `%result = pto.vshr %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VSHR`
- semantics:
  Shifts each lane of `%lhs` right by the amount carried in the corresponding `%rhs` lane.
- CCE correspondence:
  `vshr(...)`
  `__builtin_cce_vshr_*`

## 8. Vec-Scalar Ops

### `pto.vmuls`

- syntax:
  `%result = pto.vmuls %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `%scalar` is the scalar operand applied to every lane, and `%result` is the computed vector result.
- ISA family:
  `VMULS`
- semantics:
  Multiplies each input lane by the scalar operand.
- CCE correspondence:
  `vmuls(...)`
  `__builtin_cce_vmuls_*`

### `pto.vadds`

- syntax:
  `%result = pto.vadds %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `%scalar` is the scalar operand applied to every lane, and `%result` is the computed vector result.
- ISA family:
  `VADDS`
- semantics:
  Adds the scalar operand to each input lane.
- CCE correspondence:
  `vadds(...)`
  `__builtin_cce_vadds_*`

### `pto.vmaxs`

- syntax:
  `%result = pto.vmaxs %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `%scalar` is the scalar operand applied to every lane, and `%result` is the computed vector result.
- ISA family:
  `VMAXS`
- semantics:
  Computes the lane-wise maximum of the input vector and the scalar operand.
- CCE correspondence:
  `vmaxs(...)`
  `__builtin_cce_vmaxs_*`

### `pto.vmins`

- syntax:
  `%result = pto.vmins %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `%scalar` is the scalar operand applied to every lane, and `%result` is the computed vector result.
- ISA family:
  `VMINS`
- semantics:
  Computes the lane-wise minimum of the input vector and the scalar operand.
- CCE correspondence:
  `vmins(...)`
  `__builtin_cce_vmins_*`

### `pto.vlrelu`

- syntax:
  `%result = pto.vlrelu %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `%scalar` is the scalar operand applied to every lane, and `%result` is the computed vector result.
- ISA family:
  `VLRELU`
- semantics:
  Applies a leaky-ReLU style lane-wise transform using the scalar slope operand.
- CCE correspondence:
  `vlrelu(...)`
  `__builtin_cce_vlrelu_*`

### `pto.vshls`

- syntax:
  `%result = pto.vshls %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `%scalar` is the scalar operand applied to every lane, and `%result` is the computed vector result.
- ISA family:
  `VSHLS`
- semantics:
  Shifts each input lane left by the scalar shift amount.
- CCE correspondence:
  `vshls(...)`
  `__builtin_cce_vshls_*`

### `pto.vshrs`

- syntax:
  `%result = pto.vshrs %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `%scalar` is the scalar operand applied to every lane, and `%result` is the computed vector result.
- ISA family:
  `VSHRS`
- semantics:
  Shifts each input lane right by the scalar shift amount.
- CCE correspondence:
  `vshrs(...)`
  `__builtin_cce_vshrs_*`

## 9. Carry, Compare And Select

ISA assertions for this family:

- Predicate-gated arithmetic uses the supplied predicate as the active-lane mask.
- Comparison results are predicates, not integer vectors.
- Comparison families use zeroing semantics for inactive destination lanes.


### `pto.vaddc`

- syntax:
  `%result, %carry = pto.vaddc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- operand roles:
  `%lhs` and `%rhs` are the source vectors, `%mask` is the predicate control, `%result` is the arithmetic result, and `%carry` is the produced carry or borrow predicate.
- ISA family:
  `VADDC`
- semantics:
  For each lane enabled by `%mask`, adds `%lhs` and `%rhs`, writes the arithmetic result to `%result`, and writes the lane carry-out bit to `%carry`.
- CCE correspondence:
  `vaddc(...)`
  `__builtin_cce_vaddc_*`

### `pto.vsubc`

- syntax:
  `%result, %carry = pto.vsubc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- operand roles:
  `%lhs` and `%rhs` are the source vectors, `%mask` is the predicate control, `%result` is the arithmetic result, and `%carry` is the produced carry or borrow predicate.
- ISA family:
  `VSUBC`
- semantics:
  For each lane enabled by `%mask`, subtracts `%rhs` from `%lhs`, writes the arithmetic result to `%result`, and writes the lane carry-or-borrow bit to `%carry`.
- CCE correspondence:
  `vsubc(...)`
  `__builtin_cce_vsubc_*`

### `pto.vaddcs`

- syntax:
  `%result, %carry = pto.vaddcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- operand roles:
  `%lhs` and `%rhs` are the source vectors, `%carry_in` is the incoming carry or borrow predicate, `%mask` is the predicate control, `%result` is the arithmetic result, and `%carry` is the updated carry or borrow predicate.
- ISA family:
  `VADDCS`
- semantics:
  For each lane enabled by `%mask`, adds `%lhs`, `%rhs`, and the carry-in bit from `%carry_in`, writes the arithmetic result to `%result`, and writes the successor carry bit to `%carry`.
- CCE correspondence:
  `vaddcs(...)`
  `__builtin_cce_vaddcs_*`

### `pto.vsubcs`

- syntax:
  `%result, %carry = pto.vsubcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- operand roles:
  `%lhs` and `%rhs` are the source vectors, `%carry_in` is the incoming carry or borrow predicate, `%mask` is the predicate control, `%result` is the arithmetic result, and `%carry` is the updated carry or borrow predicate.
- ISA family:
  `VSUBCS`
- semantics:
  For each lane enabled by `%mask`, subtracts `%rhs` and the carry-or-borrow bit from `%carry_in` from `%lhs`, writes the arithmetic result to `%result`, and writes the successor carry-or-borrow bit to `%carry`.
- CCE correspondence:
  `vsubcs(...)`
  `__builtin_cce_vsubcs_*`

### `pto.vsel`

- syntax:
  `%result = pto.vsel %src0, %src1, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- operand roles:
  `%src0` and `%src1` are the candidate source vectors, `%mask` selects which source each lane takes, and `%result` is the selected vector.
- ISA family:
  `VSEL`
- semantics:
  Selects per lane between `%src0` and `%src1` under the control predicate `%mask`.
- CCE correspondence:
  `vsel(...)`
  `__builtin_cce_vsel_*`

### `pto.vselr`

- syntax:
  `%result = pto.vselr %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>`
- operand roles:
  `%src0` is the data vector, `%src1` is the integer lane-selector vector, and `%result` is the selected or permuted output vector.
- ISA family:
  `VSELR`
- semantics:
  Selects or permutes lanes from `%src0` using the lane indices carried in `%src1`.
- CCE correspondence:
  `vselr(...)`
  `__builtin_cce_vselr_*`

### `pto.vselrv2`

- syntax:
  `%result = pto.vselrv2 %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>`
- operand roles:
  `%src0` is the data vector, `%src1` is the integer lane-selector vector, and `%result` is the selected or permuted output vector.
- ISA family:
  `VSELR v2`
- semantics:
  Selects or permutes lanes from `%src0` using the lane indices carried in `%src1`.
- CCE correspondence:
  `vselrv2(...)`
  `__builtin_cce_vselrv2_*`

### `pto.vcmp`

- syntax:
  `%result = pto.vcmp %src0, %src1, %mask, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.mask`
- operand roles:
  `%src0` and `%src1` are the values being compared, `%mask` is the seed predicate or enable mask, `"CMP_MODE"` selects the comparison relation, and `%result` is the produced predicate.
- ISA family:
  `VCMP`
- semantics:
  For each lane enabled by `%mask`, compares `%src0` and `%src1` using `CMP_MODE` and writes the boolean result into `%result`. Lanes disabled by `%mask` are cleared to zero in the returned predicate.
- CCE correspondence:
  `vcmp(...)`
  `__builtin_cce_vcmp_<op>_*_z`

### `pto.vcmps`

- syntax:
  `%result = pto.vcmps %src, %scalar, %mask, "CMP_MODE" : !pto.vreg<NxT>, T, !pto.mask -> !pto.mask`
- operand roles:
  `%src` is the vector input, `%scalar` is the scalar comparison value, `%mask` is the seed predicate or enable mask, `"CMP_MODE"` selects the comparison relation, and `%result` is the produced predicate.
- ISA family:
  `VCMPS`
- semantics:
  For each lane enabled by `%mask`, compares `%src` against `%scalar` using `CMP_MODE` and writes the boolean result into `%result`. Lanes disabled by `%mask` are cleared to zero in the returned predicate.
- CCE correspondence:
  `vcmps(...)`
  `__builtin_cce_vcmps_<op>_*_z`

### `pto.vpnot`

- syntax:
  `%result = pto.vpnot %input, %mask : !pto.mask, !pto.mask -> !pto.mask`
- operand roles:
  `%input` is the source predicate, `%mask` is the predicate control, and `%result` is the inverted predicate result.
- ISA family:
  `PNOT`
- semantics:
  For each lane enabled by `%mask`, inverts the corresponding bit of `%input` and writes the result bit to `%result`.
- CCE correspondence:
  `pnot(...)`

### `pto.vpsel`

- syntax:
  `%result = pto.vpsel %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- operand roles:
  `%src0` and `%src1` are the candidate source predicates, `%mask` selects which predicate each bit takes, and `%result` is the selected predicate.
- ISA family:
  `PSEL`
- semantics:
  Selects each result predicate bit from `%src0` or `%src1` under the control of `%mask`.
- CCE correspondence:
  `psel(...)`

## 10. Pairing And Interleave

### `pto.vpdintlv_b8`

- syntax:
  `%low, %high = pto.vpdintlv_b8 %lhs, %rhs : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- operand roles:
  `%lhs` and `%rhs` are the two source predicates, and `%low` plus `%high` are the two predicate results produced by the interleave or deinterleave split.
- ISA family:
  `PDINTLV`
- semantics:
  Deinterleaves predicate data into low and high predicate results.
- CCE correspondence:
  predicate interleave/deinterleave family

### `pto.vpintlv_b16`

- syntax:
  `%low, %high = pto.vpintlv_b16 %lhs, %rhs : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- operand roles:
  `%lhs` and `%rhs` are the two source predicates, and `%low` plus `%high` are the two predicate results produced by the interleave or deinterleave split.
- ISA family:
  `PINTLV`
- semantics:
  Interleaves predicate data into low and high predicate results.
- CCE correspondence:
  predicate interleave/deinterleave family

### `pto.vintlv`

- syntax:
  `%low, %high = pto.vintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- operand roles:
  `%lhs` and `%rhs` are the two source vectors, and `%low` plus `%high` are the two vector results produced by the interleave or deinterleave split.
- ISA family:
  `VINTLV`
- semantics:
  Interleaves lanes from `%lhs` and `%rhs` into one combined lane stream and returns the low half in `%low` and the high half in `%high`.
- CCE correspondence:
  `vintlv(...)`
  `__builtin_cce_vintlv_*`

### `pto.vdintlv`

- syntax:
  `%low, %high = pto.vdintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- operand roles:
  `%lhs` and `%rhs` are the two source vectors, and `%low` plus `%high` are the two vector results produced by the interleave or deinterleave split.
- ISA family:
  `VDINTLV`
- semantics:
  Deinterleaves the combined lane streams in `%lhs` and `%rhs` and returns the low deinterleaved half in `%low` and the high deinterleaved half in `%high`.
- CCE correspondence:
  `vdintlv(...)`
  `__builtin_cce_vdintlv_*`

### `pto.vintlvv2`

- syntax:
  `%result = pto.vintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` and `%rhs` are the two source vectors, `"PART"` selects which half of the interleaved lane stream is returned, and `%result` is the selected vector result.
- ISA family:
  `VINTLV v2`
- semantics:
  Interleaves `%lhs` and `%rhs` into one combined lane stream and returns only the half selected by `PART`.
- CCE correspondence:
  `vintlvv2(...)`
  `__builtin_cce_vintlvv2_*`

### `pto.vdintlvv2`

- syntax:
  `%result = pto.vdintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` and `%rhs` are the two source vectors, `"PART"` selects which half of the deinterleaved lane stream is returned, and `%result` is the selected vector result.
- ISA family:
  `VDINTLV v2`
- semantics:
  Deinterleaves `%lhs` and `%rhs` into one logical lane stream and returns only the half selected by `PART`.
- CCE correspondence:
  `vdintlvv2(...)`
  `__builtin_cce_vdintlvv2_*`

## 11. Conversion, Index And Sort

ISA assertions for this family:

- For width-changing conversions, predication is applied to input lanes and is composed with part-selection controls such as even/odd or packed-part selectors.
- Narrowing conversions place results into the selected destination part and zero the remaining part of the widened slot.
- Widening conversions read only the selected source part; the unselected part is architecturally ignored.
- Sort families operate on Vector tile buffer-resident proposal data and preserve the ISA tie-break rule that lower original indices win on equal scores.


### `pto.vtrc`

- syntax:
  `%result = pto.vtrc %input, "ROUND_MODE" : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `"ROUND_MODE"` selects the rounding behavior, and `%result` is the converted vector result.
- ISA family:
  `VTRC`
- semantics:
  Rounds each input lane according to `ROUND_MODE` and returns the truncated converted result. The rounding mode is part of the ISA-visible semantics, not a lowering hint.
- CCE correspondence:
  `vtrc(...)`
  `__builtin_cce_vtrc_*`

### `pto.vcvt`

- syntax:
  `%result = pto.vcvt %input {round_mode = "ROUND_MODE", sat = "SAT_MODE", part = "PART_MODE"} : !pto.vreg<NxT0> -> !pto.vreg<NxT1>`
- operand roles:
  `%input` is the source vector, `"ROUND_MODE"` selects the rounding behavior, `"SAT_MODE"` selects saturation or truncation behavior, `"PART_MODE"` selects the even or odd conversion part when required by the ISA form, and `%result` is the converted vector result.
- ISA family:
  `VCVTFI` / `VCVTFF` / `VCVTIF` / `VCVTII`
- semantics:
  Converts `%input` lane-wise according to the source type, destination type, rounding rule, saturation rule, and part-selection rule encoded by the op form. Width-changing forms consume only the selected source part or produce only the selected destination part exactly as required by the ISA conversion family.
- CCE correspondence:
  `vcvt(...)`
  builtin families:
  `__builtin_cce_vcvt*`, `__builtin_cce_vcvtfi_*`, `__builtin_cce_vcvtif_*`, `__builtin_cce_vcvtii_*`, `__builtin_cce_vcvtff_*`

### `pto.vci`

- syntax:
  `%result = pto.vci %index {order = "ORDER"} : integer -> !pto.vreg<NxT>`
- operand roles:
  `%index` is the scalar seed or base index value, `"ORDER"` selects the lane-index ordering policy, and `%result` is the generated integer index vector.
- ISA family:
  `VCI`
- semantics:
  Materializes lane indices from the scalar seed value using the selected ordering policy.
- CCE correspondence:
  `vci(...)`
  `__builtin_cce_vci_*`

### `pto.vbitsort`

- syntax:
  `pto.vbitsort %destination, %source, %indices, %repeat_times : !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, index`
- operand roles:
  `%destination` is the Vector tile buffer output buffer, `%source` is the Vector tile buffer score buffer, `%indices` is the Vector tile buffer index buffer, and `%repeat_times` is the repeat count for consecutive sort invocations.
- ISA family:
  `VBS32`
- semantics:
  Sorts the proposal records named by `%source` and `%indices` and writes the ordered result stream to `%destination`. `repeat_times` controls how many consecutive sort iterations are performed, equal scores are ordered by lower original proposal index first, and the destination region must not overlap the source regions.
- CCE correspondence:
  `vbitsort(...)`
  `__builtin_cce_vbitsort_*`

### `pto.vmrgsort4`

- syntax:
  `pto.vmrgsort4 %destination, %source0, %source1, %source2, %source3, %count, %config : !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64`
- operand roles:
  `%destination` is the Vector tile buffer output buffer, `%source0` through `%source3` are the four Vector tile buffer input list bases, `%count` is the total work or encoded list-count payload, and `%config` is the ISA merge-sort configuration word.
- ISA family:
  `VMS4v2`
- semantics:
  Merges four sorted proposal lists from the Vector tile buffer into one sorted output stream. On equal scores, entries from the lower-numbered input list win; `%config` controls repeat and exhausted-input behavior; and source and destination regions must satisfy the ISA non-overlap rules.
- CCE correspondence:
  `vmrgsort4(...)`
  `__builtin_cce_vmrgsort4_*`

## 12. Extended Arithmetic

### `pto.vmull`

- syntax:
  `%low, %high = pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- operand roles:
  `%lhs` and `%rhs` are the source vectors, `%mask` is the predicate control, and `%low` plus `%high` are the split widened product results.
- ISA family:
  `VMULL`
- semantics:
  Multiplies lanes under `%mask` and returns the widened product split across low and high result vectors.
- CCE correspondence:
  `vmull(...)`
  `__builtin_cce_vmull_*`

### `pto.vmula`

- syntax:
  `%result = pto.vmula %acc, %lhs, %rhs, %mask {mode = "MODE"} : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- operand roles:
  `%acc` is the accumulator input, `%lhs` and `%rhs` are the multiplicands, `%mask` is the predicate control, `"MODE"` selects merging or zeroing behavior, and `%result` is the accumulated vector result.
- ISA family:
  `VMULA`
- semantics:
  Performs masked vector multiply-accumulate into `%acc`, with `mode` controlling the merge or zeroing behavior.
- CCE correspondence:
  `vmula(...)`
  `__builtin_cce_vmula_*_m`

## 13. Stateless Stores

ISA assertions for this family:

- These ops write to Vector tile buffer-backed storage and must satisfy the distribution-specific destination-alignment rules defined by the ISA family.
- ISA forms that can post-update a shared register are represented here only as addressed stores; hidden pointer mutation is not part of these stateless forms.
- Predicate-store data is architecturally `b8`, regardless of the scalar element type used by surrounding vector code.


### `pto.vsts`

- syntax:
  `pto.vsts %value, %destination[%offset] {dist = "DIST"} : !pto.vreg<NxT>, !llvm.ptr<AS>`
- operand roles:
  `%value` is the vector being stored, `%destination` is the Vector tile buffer base pointer, `%offset` is the store displacement, and `"DIST"` selects the ISA store distribution mode.
- ISA family:
  `VST` / `VSTI` / `VSTS`
- semantics:
  Stores `%value` to the Vector tile buffer at `destination + offset` using the store form selected by `DIST`. The distribution token determines the destination lane layout and alignment requirement of the access.
- CCE correspondence:
  `vst(...)`, `vsts(...)`
  `__builtin_cce_vstx1_*`, `__builtin_cce_vstsx1_*`

### `pto.vscatter`

- syntax:
  `pto.vscatter %value, %destination, %offsets, %active_lanes : !pto.vreg<NxT>, !llvm.ptr<AS>, !pto.vreg<NxI>, index`
- operand roles:
  `%value` is the vector being scattered, `%destination` is the Vector tile buffer base pointer, `%offsets` is the per-lane offset vector, and `%active_lanes` bounds how many lanes participate.
- ISA family:
  `VSCATTER`
- semantics:
  Scatters active vector lanes to Vector tile buffer addresses derived from `%destination` and `%offsets`.
- CCE correspondence:
  `vscatter(...)`
  `__builtin_cce_vscatter_*`

### `pto.vsts_pred`

- syntax:
  `pto.vsts_pred %value, %destination[%offset], %active_lanes {dist = "DIST"} : !pto.vreg<NxT>, !llvm.ptr<AS>, index`
- operand roles:
  `%value` is the vector being stored, `%destination` is the Vector tile buffer base pointer, `%offset` is the store displacement, `%active_lanes` bounds the active prefix, and `"DIST"` selects the ISA store distribution mode.
- ISA family:
  `Predicated vector-store helper family`
- semantics:
  Stores only the active prefix of `%value` selected by `%active_lanes`, using the layout and alignment rules implied by `DIST`. Lanes outside the active prefix do not update memory.
- CCE correspondence:
  predicated vector store family

### `pto.vpsts`

- syntax:
  `pto.vpsts %value, %destination[%offset] : !pto.mask, !llvm.ptr<AS>`
- operand roles:
  `%value` is the predicate being stored, `%destination` is the Vector tile buffer base pointer, and `%offset` is the store displacement.
- ISA family:
  `PSTS`
- semantics:
  Stores predicate state to the Vector tile buffer at `destination + offset`.
- CCE correspondence:
  `psts(...)`
  `__builtin_cce_psts_b8`, `__builtin_cce_psts_post_b8`

### `pto.vpst`

- syntax:
  `pto.vpst %value, %destination[%offset], "DIST" : !pto.mask, !llvm.ptr<AS>, index`
- operand roles:
  `%value` is the predicate being stored, `%destination` is the Vector tile buffer base pointer, `%offset` is the store displacement, and `"DIST"` selects the predicate-store distribution token.
- ISA family:
  `PST`
- semantics:
  Stores predicate state to the Vector tile buffer using an explicit index offset and predicate-store distribution token.
- CCE correspondence:
  `pst(...)`
  `__builtin_cce_pst_b8`

### `pto.vpsti`

- syntax:
  `pto.vpsti %value, %destination, %offset, "DIST" : !pto.mask, !llvm.ptr<AS>, i32`
- operand roles:
  `%value` is the predicate being stored, `%destination` is the Vector tile buffer base pointer, `%offset` is the scalar displacement, and `"DIST"` selects the predicate-store distribution token.
- ISA family:
  `PSTI`
- semantics:
  Stores predicate state to the Vector tile buffer using an immediate-style scalar offset and predicate-store distribution token.
- CCE correspondence:
  `psti(...)`
  `__builtin_cce_psti_b8`, `__builtin_cce_psti_post_b8`

### `pto.vsst`

- syntax:
  `pto.vsst %value, %destination[%offset], "STRIDE" : !pto.vreg<NxT>, !llvm.ptr<AS>`
- operand roles:
  `%value` is the vector being stored, `%destination` is the Vector tile buffer base pointer, `%offset` is the store displacement, and `"STRIDE"` selects the ISA strided-store token.
- ISA family:
  `VSST`
- semantics:
  Stores vector data to the Vector tile buffer using a stride token rather than a regular contiguous distribution.
- CCE correspondence:
  `vsst(...)`
  `__builtin_cce_vsst_*`

### `pto.vstx2`

- syntax:
  `pto.vstx2 %low, %high, %destination[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !llvm.ptr<AS>, index, !pto.mask`
- operand roles:
  `%low` and `%high` are the two source vectors being stored, `%destination` is the Vector tile buffer base pointer, `%offset` is the store displacement, `"DIST"` selects the x2 store distribution token, and `%mask` is the predicate control.
- ISA family:
  `VST x2`
- semantics:
  Stores `%low` and `%high` to the Vector tile buffer as one x2 store operation. `DIST` determines how the two source vectors are interleaved into memory, and `%mask` gates which lanes update memory.
- CCE correspondence:
  `vst(...)`
  `__builtin_cce_vstx2_*`

### `pto.vsstb`

- syntax:
  `pto.vsstb %value, %destination, %offset, %mask : !pto.vreg<NxT>, !llvm.ptr<AS>, i32, !pto.mask`
- operand roles:
  `%value` is the vector being stored, `%destination` is the Vector tile buffer base pointer, `%offset` is the scalar displacement, and `%mask` is the predicate control.
- ISA family:
  `VSSTB`
- semantics:
  Performs a masked strided vector store using a scalar offset and predicate mask.
- CCE correspondence:
  `vsstb(...)`
  `__builtin_cce_vsstb_*`, `__builtin_cce_vsstb_post_*`

### `pto.vsta`

- syntax:
  `pto.vsta %value, %destination[%offset] : !pto.align, !llvm.ptr<AS>, index`
- operand roles:
  `%value` is the align payload being stored, `%destination` is the Vector tile buffer base pointer, and `%offset` is the store displacement.
- ISA family:
  `VSTA`
- semantics:
  Stores align-state payload to the Vector tile buffer at `destination + offset`.
- CCE correspondence:
  `vsta(...)`
  `__builtin_cce_vsta_*`

### `pto.vstas`

- syntax:
  `pto.vstas %value, %destination, %offset : !pto.align, !llvm.ptr<AS>, i32`
- operand roles:
  `%value` is the align payload being stored, `%destination` is the Vector tile buffer base pointer, and `%offset` is the scalar displacement.
- ISA family:
  `VSTAS`
- semantics:
  Stores align-state payload to the Vector tile buffer using a scalar offset form.
- CCE correspondence:
  `vstas(...)`
  `__builtin_cce_vstas_*`, `__builtin_cce_vstas_post_*`

### `pto.vstar`

- syntax:
  `pto.vstar %value, %destination : !pto.align, !llvm.ptr<AS>`
- operand roles:
  `%value` is the align payload being stored and `%destination` is the base pointer used by the register-update store form.
- ISA family:
  `VSTAR`
- semantics:
  Stores align-state payload using the base pointer carried directly by `%destination`.
- CCE correspondence:
  `vstar(...)`
  `__builtin_cce_vstar_*`

## 14. Stateful Store Ops

ISA assertions for this family:

- These ops expose the ISA's hidden align-register and address-update effects as explicit SSA results.
- `"MODE"` determines whether the underlying ISA performs post-update or preserves the incoming base state.
- Correct programs thread the returned align or base state into the next dependent stateful store on the same logical stream.


These ops make ISA reference-updated state explicit as SSA results.

### `pto.vpstu`

- syntax:
  `%align_out, %base_out = pto.vpstu %align_in, %value, %base : !pto.align, !pto.mask, !llvm.ptr<AS> -> !pto.align, !llvm.ptr<AS>`
- operand roles:
  `%align_in` is the incoming align state, `%value` is the predicate being stored, `%base` is the current base pointer, `%align_out` is the updated align state, and `%base_out` is the updated base pointer.
- ISA family:
  `PSTU`
- semantics:
  Stores predicate data through the stateful predicate-store form and returns the align state and base pointer state after the store. The returned state is the exact architected successor state for the next dependent store in the same stream.
- CCE correspondence:
  `pstu(...)`
  `__builtin_cce_pstu_b16`, `__builtin_cce_pstu_b32`

### `pto.vstu`

- syntax:
  `%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base, "MODE" : !pto.align, index, !pto.vreg<NxT>, !llvm.ptr<AS> -> !pto.align, index`
- operand roles:
  `%align_in` is the incoming align state, `%offset_in` is the current index displacement, `%value` is the vector being stored, `%base` is the current base pointer, `"MODE"` selects post-update behavior, `%align_out` is the updated align state, and `%offset_out` is the updated index displacement.
- ISA family:
  `VSTU`
- semantics:
  Stores `%value` through the stateful unaligned-store form addressed by `%base` and `%offset_in`, then returns the successor align state and successor index displacement. If `MODE` is `POST_UPDATE`, `%offset_out` is the ISA-updated displacement; if `MODE` is `NO_POST_UPDATE`, `%offset_out` preserves the incoming displacement.
- CCE correspondence:
  `vstu(...)`
  `__builtin_cce_vstu_*`

### `pto.vstus`

- syntax:
  `%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base, "MODE" : !pto.align, i32, !pto.vreg<NxT>, !llvm.ptr<AS> -> !pto.align, !llvm.ptr<AS>`
- operand roles:
  `%align_in` is the incoming align state, `%offset` is the scalar displacement, `%value` is the vector being stored, `%base` is the current base pointer, `"MODE"` selects post-update behavior, `%align_out` is the updated align state, and `%base_out` is the updated base pointer.
- ISA family:
  `VSTUS`
- semantics:
  Stores `%value` through the scalar-offset stateful unaligned-store form and returns the successor align state and successor base pointer. If `MODE` is `POST_UPDATE`, `%base_out` is the ISA-updated base; if `MODE` is `NO_POST_UPDATE`, `%base_out` preserves the incoming base.
- CCE correspondence:
  `vstus(...)`
  `__builtin_cce_vstus_*`, `__builtin_cce_vstus_post_*`

### `pto.vstur`

- syntax:
  `%align_out = pto.vstur %align_in, %value, %base, "MODE" : !pto.align, !pto.vreg<NxT>, !llvm.ptr<AS> -> !pto.align`
- operand roles:
  `%align_in` is the incoming align state, `%value` is the vector being stored, `%base` is the current base pointer, `"MODE"` selects post-update behavior, and `%align_out` is the updated align state.
- ISA family:
  `VSTUR`
- semantics:
  Stores `%value` through the register-update stateful unaligned-store form and returns the successor align state. Any architected base update is controlled by `MODE` and is not hidden from surrounding VPTO state threading.
- CCE correspondence:
  `vstur(...)`
  `__builtin_cce_vstur_*`

### Chained Usage Example

Stateful store ops make the implicit ISA update chain explicit in SSA form.
A typical sequence starts from an align-producing load-side op such as
`pto.vldas`, then threads the returned align or base values through each store.

```mlir
%align0 = pto.vldas %src[%c0] : !llvm.ptr<6> -> !pto.align
%align1, %offset1 = pto.vstu %align0, %c0, %value0, %dst, "POST_UPDATE"
    : !pto.align, index, !pto.vreg<64xf32>, !llvm.ptr<6> -> !pto.align, index
%align2, %base1 = pto.vstus %align1, %c32_i32, %value1, %dst, "POST_UPDATE"
    : !pto.align, i32, !pto.vreg<64xf32>, !llvm.ptr<6> -> !pto.align, !llvm.ptr<6>
%align3 = pto.vstur %align2, %value2, %base1, "NO_POST_UPDATE"
    : !pto.align, !pto.vreg<64xf32>, !llvm.ptr<6> -> !pto.align
```

In this form, VPTO makes the ordering and the address-state evolution visible to
verification and later lowering passes instead of leaving them as hidden side
effects on an implicit alignment register or base pointer.

## 15. Vector Thread And Loop Control

ISA assertions for this family:

- Vector-thread control ops execute in the scalar control domain but define
  architectural state consumed by later vector-thread execution.
- `instr_num` fields count the loop-body or fetch-body instructions following
  the control op; they MUST match the architecturally executed body size.
- Loop-control forms MUST preserve the ordering, loop-layer, and once-only
  restrictions stated below; these are correctness rules, not performance
  hints.

### `pto.vthread_fetch`

- covered ISA mnemonics:
  `VFI`, `VF`, `VFI_RU`, `VFRU`, `VFI_BC`, `VF_BC`, `VFI_PREFETCH`,
  `VF_PREFETCH`
- syntax:
  `pto.vthread_fetch %target, %instr_count {target_mode = "pc_rel|absolute", action = "execute|prefetch", pbid = %pbid?, compat_template = i2?, compat_block_stride = true|false} : T, i16`
- operand roles:
  `%target` is either the relative vector-PC delta or the absolute vector-PC
  start address, `%instr_count` is the unsigned instruction count of the fetched
  vector body, `pbid` names the parameter-buffer resource attached to RU forms,
  `compat_template` selects the backward-compatible PB-to-SREG mapping template,
  and `compat_block_stride` enables the legacy block-stride PB mapping.
- data types:
  `%target` is a signed 16-bit PC-relative displacement for `VFI*` forms and a
  scalar address value for `VF*` forms. `%instr_count` is an unsigned 16-bit
  count.
- semantics:
  Starts a vector-thread fetch window or prefetch window. For `pc_rel` forms,
  the fetched vector PC equals `sign_extend(target * 4) + PC_of_main_scalar`.
  For `absolute` forms, the fetched vector PC equals `%target`. `action =
  "prefetch"` performs instruction prefetch only and MUST NOT execute the
  fetched body. RU forms associate the fetch with `PBID`. BC forms apply the
  compatibility template to map PB entries and, when enabled, the legacy
  block-stride extension fields into shared-register state.
- assertions and exceptions:
  One fetch form launches or prefetches exactly one vector body. The PC-relative
  base is the first 4-byte slot of the fetch instruction, even when the control
  instruction occupies multiple slots.

### `pto.vparam_buffer`

- covered ISA mnemonics:
  `PUSH_PB`, `RELEASE_PBID`
- syntax:
  `pto.vparam_buffer "push" %xd, %xn, %xm, %xt : T, T, T, T`
  `pto.vparam_buffer "release" %pbid : T`
- operand roles:
  `%xd`, `%xn`, `%xm`, and `%xt` are the four scalar words pushed into the
  parameter buffer in order; `%pbid` is the parameter-buffer identifier being
  released.
- data types:
  Push consumes four scalar register-width words and writes one 256-bit PB
  entry. Release consumes one scalar resource identifier.
- semantics:
  `push` writes `{%xt, %xm, %xn, %xd}` into the next vector-thread parameter
  buffer entry. `release` returns a previously used parameter-buffer identifier
  to the implementation so it can be reused by a later vector loop.

### `pto.vloop`

- covered ISA mnemonics:
  `VLOOPv2`, `VLOOPN`
- syntax:
  `pto.vloop %bound_or_count, %instr_count {mode = "counted|elements", element_type = "b8|b16|b32"?, layer = i4?, last = true|false?, loop_index = "i1|i2|i3|i4"?} : T, i16`
- operand roles:
  `%bound_or_count` is the loop trip count for counted loops and the total
  element count `N` for element-counted loops, `%instr_count` is the loop-body
  instruction count, `layer` and `last` describe the nesting layer for counted
  loops, and `loop_index` selects which hardware loop level receives an
  element-counted bound.
- data types:
  Counted loops consume a 16-bit unsigned loop count. Element-counted loops
  consume a positive total element count and one of `b8`, `b16`, or `b32` as
  the element granularity.
- semantics:
  `mode = "counted"` begins a hardware vector loop whose iteration count equals
  `%bound_or_count`; if the count is zero the loop body is skipped. `mode =
  "elements"` derives the loop count from the total element count `N` as
  `(N - 1) / VL_t` and defines a final partial iteration in which only the
  lowest `K = (N % VL_t == 0) ? VL_t : (N % VL_t)` elements are active.
- assertions and exceptions:
  `%instr_count` MUST be non-zero. The `layer` / `last` fields MUST encode a
  valid nesting topology. The bound register used by a counted loop MUST NOT be
  post-updated inside that loop body. `N` in an element-counted loop MUST be
  greater than zero.

### `pto.vpred_decl`

- covered ISA mnemonics:
  `VPD`
- syntax:
  `pto.vpred_decl %predicate : !pto.mask`
- operand roles:
  `%predicate` names the predicate register that hardware auto-updates for the
  surrounding element-counted loop.
- data types:
  `%predicate` is a predicate register at the granularity implied by the
  associated `VLOOPN` element type.
- semantics:
  Declares the predicate register updated by the active `VLOOPN` layer. For all
  full iterations the declared predicate is all-active; on the final partial
  iteration only the lowest `K` elements remain active.
- assertions and exceptions:
  Each element-counted loop can have at most one architecturally effective
  predicate declaration. If multiple declarations are emitted, only the last
  one in the loop body is effective and earlier ones are semantically dead.

### `pto.vloop_exit`

- covered ISA mnemonics:
  `VEXT`, `VEXTFA`, `SEXT`
- syntax:
  `pto.vloop_exit %condition {condition_kind = "pred0|cmp", flush_align = true|false, compare_type = "s16|u16|s32|u32"?, compare_mode = "eq|ne|lt|gt|le|ge"?} : T`
- operand roles:
  `%condition` is either the governing predicate register or the pair of shared
  registers compared by `SEXT`, `flush_align` requests the alignment-register
  flush behavior of `VEXTFA`, and `compare_type` / `compare_mode` define the
  typed comparison performed by `SEXT`.
- data types:
  Predicate exits consume a predicate register and test its least-significant
  bit. Compare exits consume typed `s16`, `u16`, `s32`, or `u32` shared-register
  values.
- semantics:
  Schedules an exit from every enclosing hardware loop layer that contains the
  exit instruction after all remaining instructions of the current iteration
  execute. `VEXT` tests the first predicate bit. `VEXTFA` performs the same exit
  and additionally flushes every dirty alignment register to the address implied
  by its last unaligned-store post-update state. `SEXT` exits when the typed
  comparison between its operands is true.
- assertions and exceptions:
  An exit op MUST be the last instruction in its loop layer and there can be at
  most one exit op in a given layer. If unaligned-store state remains dirty and
  the program exits without a flushing form, the pending tail bytes remain the
  programmer's responsibility.

### `pto.vaddr_gen`

- covered ISA mnemonics:
  `VAG`
- syntax:
  `pto.vaddr_gen %ad, %i1_stride, %i2_stride, %i3_stride, %i4_stride : T, T, T, T, T`
- operand roles:
  `%ad` is the destination address-register state, and the four stride operands
  are the loop-layer contributions for `i1` through `i4`.
- data types:
  The stride operands are unsigned shared-register values interpreted in address
  units for the corresponding loop layer.
- semantics:
  Defines the affine address evolution used by vector load/store instructions:
  `Ad = const1*i1 + const2*i2 + const3*i3 + const4*i4`, omitting absent outer
  layers. When multiple declarations target the same address register, the last
  declaration overwrites earlier ones.
- assertions and exceptions:
  `VAG` MUST be outside `VLOOPv2` bodies. When the address register is also used
  by post-update load/store forms, the resulting address stream MUST remain
  consecutive across loop boundaries.

### `pto.vspr_store`

- covered ISA mnemonics:
  `SPRSTI`, `SPRSTS`, `SPRCLR`
- syntax:
  `pto.vspr_store "store_imm" %spr, %base, %offset {post_update = true|false}`
  `pto.vspr_store "store_reg" %spr, %base, %offset_reg {post_update = true|false}`
  `pto.vspr_store "clear" %spr`
- operand roles:
  `%spr` is the special register being stored or cleared, `%base` is the base
  address shared register, `%offset` is the signed immediate displacement,
  `%offset_reg` is the signed byte displacement from a shared register, and
  `post_update` selects whether the base register is updated after the store.
- data types:
  `%base` is a shared-register address value, `%offset` is a signed 8-bit
  immediate scaled by the stored-SPR width, and `%offset_reg` is a signed
  32-bit byte displacement.
- semantics:
  Stores the selected SPR to memory at the architecturally addressed location or
  clears the SPR to zero. If `post_update` is set, the base register itself is
  used as the store address and then advanced by the scaled immediate or by the
  register offset.
- assertions and exceptions:
  The access address MUST satisfy the alignment requirement of the named SPR.
  Current ISA text only admits `AR` as the stored or cleared SPR in this family.

### `pto.vmove_ub`

- covered ISA mnemonics:
  `MOV_UB`
- syntax:
  `pto.vmove_ub %dst, %src, %config : T, T, T`
- operand roles:
  `%dst` and `%src` are Vector tile buffer addresses, and `%config` is the ISA-defined control
  word that determines the transfer submode.
- data types:
  `%dst` and `%src` are Vector tile buffer address operands. `%config` is a family-specific
  scalar control word.
- semantics:
  Performs the Vector-tile-buffer-to-Vector-tile-buffer movement defined by the configuration word. VPTO treats
  the addressing, element grouping, and submode fields as part of the
  architectural contract of `%config`; no encoding bits are elided.
- assertions and exceptions:
  This family remains configuration-driven. Any frontend that lowers to this
  VPTO family MUST preserve the full control word rather than rewriting it into
  a narrower helper sequence.

## 16. Vector And Predicate Rearrangement

ISA assertions for this family:

- Rearrangement families do not change the arithmetic value of a source element
  unless the family explicitly states truncation, extension, packing, or
  zero-fill.
- Predicated move and predicate-logic forms follow the ISA predication mode
  stated by the family; merge and zeroing behavior are semantically distinct.

### `pto.vmov`

- covered ISA mnemonics:
  `VMOV`, `PMOV`
- syntax:
  `pto.vmov %dst, %src {domain = "vector|predicate", element_type = "b8|b16|b32"?, predicated = true|false, pred_mode = "merging|zeroing"}`
- operand roles:
  `%dst` is the destination vector or predicate register, `%src` is the source
  register, `predicated` selects whether a governing predicate is present, and
  `pred_mode` records whether inactive destination elements are preserved or
  zeroed.
- data types:
  Vector moves use `b8`, `b16`, or `b32` lane interpretation. Predicate moves
  always operate on `b8` predicate elements.
- semantics:
  Copies active source elements to the corresponding destination positions.
  Unpredicated vector moves copy all elements. Predicated vector moves are an
  alias of lane-select with merging behavior. Predicate moves are aliases of
  `PAND` with identical source operands; the predicated form uses zeroing and
  the unpredicated form uses the source predicate itself as the governing mask.

### `pto.vpack_family`

- covered ISA mnemonics:
  `VPACK`, `VZUNPACK`, `VSUNPACK`
- syntax:
  `pto.vpack_family %dst, %src {mode = "pack|zunpack|sunpack", element_type = "b8|b16|b32|s8|s16", part = "lower|higher"}`
- operand roles:
  `%dst` is the destination vector, `%src` is the source vector, `mode`
  selects pack versus zero-extension versus sign-extension, `element_type`
  selects the source lane width, and `part` selects the lower or higher half
  of the source or destination view.
- data types:
  `pack` consumes `b16` or `b32` lanes and truncates them to 8-bit or 16-bit
  packed results. `zunpack` consumes `b8` or `b16` lanes. `sunpack` consumes
  `s8` or `s16` lanes.
- semantics:
  `pack` extracts the least-significant 8-bit or 16-bit slice of each source
  lane and writes the packed results into the selected half of `%dst`, filling
  the other half with zeros. `zunpack` selects the lower or higher half of the
  source vector and zero-extends each selected lane. `sunpack` selects the same
  half and sign-extends each selected lane.

### `pto.vslide_family`

- covered ISA mnemonics:
  `VSLIDE`, `PSLIDE`, `MOVVP`
- syntax:
  `pto.vslide_family %dst, %src, %shift_or_part {domain = "vector|predicate|vector_to_predicate", element_type = "b8|b16|b32", part = i?}`
- operand roles:
  `%dst` is the destination vector or predicate register, `%src` is the source
  vector or predicate register pair base, and `%shift_or_part` is the shared
  register shift amount or extracted subvector part index.
- data types:
  Slide widths are selected from `b8`, `b16`, or `b32`. The shift operand is a
  16-bit shared-register value for slide forms and an immediate part selector
  for `MOVVP`.
- semantics:
  Vector and predicate slide forms take a right-sliding extraction window of one
  full destination width across `%src` and the next consecutive source register;
  the base source register MUST be even. `MOVVP` extracts `b16` or `b32` slices
  from a vector register, expands them to predicate-bit granularity, and writes
  the result into a predicate register.

### `pto.vpredicate_gen`

- covered ISA mnemonics:
  `PLT`, `PLTM`
- syntax:
  `pto.vpredicate_gen %dst, %limit, %loop_base? {element_type = "b8|b16|b32"}`
- operand roles:
  `%dst` is the generated predicate, `%limit` is the element-count threshold,
  and `%loop_base` is the per-iteration base index used by `PLTM`.
- data types:
  `%limit` is a 32-bit shared-register count. `%loop_base` is a 16-bit
  shared-register loop index.
- semantics:
  `PLT` sets destination element `i` true when `i < limit`, then decrements the
  shared-register count by `VL_t`, saturating the count at zero. `PLTM` sets
  destination element `i` true when `i + loop_base * VL_t < limit`. Both forms
  generate fresh predicate state rather than updating only active lanes.

### `pto.vpredicate_logic`

- covered ISA mnemonics:
  `PAND`, `POR`, `PXOR`
- syntax:
  `pto.vpredicate_logic %dst, %lhs, %rhs, %gov {logic = "and|or|xor"} : !pto.mask, !pto.mask, !pto.mask, !pto.mask`
- operand roles:
  `%dst` is the destination predicate, `%lhs` and `%rhs` are the input
  predicates, `%gov` is the governing predicate, and `logic` selects the
  bitwise operation.
- data types:
  All four operands are `!pto.mask` and are interpreted as `b8` predicate
  elements.
- semantics:
  Applies the chosen bitwise logic operation to active predicate elements only.
  The data type is always `b8`. Inactive elements are zeroed because the
  governing predicate uses zeroing predication.

### `pto.vpattern`

- covered ISA mnemonics:
  `VCP`
- syntax:
  `pto.vpattern %dst, #pat {element_type = "b16|b32"}`
- operand roles:
  `%dst` is the destination vector of integer indices and `#pat` selects the
  predefined scatter/gather pattern.
- data types:
  `%dst` uses `b16` or `b32` integer lane widths according to the selected
  pattern family.
- semantics:
  Materializes one of the ISA-defined unsigned index patterns used by layout
  transformations before gather or scatter. The pattern value fully determines
  the lane-to-lane affine formula; for example `chn4to8` maps lane `i` to
  `floor(i/4) * 8 + (i mod 4)`.

## 17. Arithmetic Extension Families

ISA assertions for this family:

- These families extend the arithmetic surface beyond the directly named
  `pto.vadd` / `pto.vsub` / `pto.vmul` / `pto.vdiv` forms without changing the
  ISA rule that inactive lanes do not contribute results.
- Where the ISA uses the destination as an accumulator input, VPTO MUST model
  that destination-read explicitly in the operation contract.

### `pto.vmod`

- covered ISA mnemonics:
  `VMOD`
- syntax:
  `pto.vmod %dst, %lhs, %rhs, %pg : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask`
- operand roles:
  `%dst` is the destination vector, `%lhs` is the dividend vector, `%rhs` is
  the divisor vector, and `%pg` is the zeroing predicate.
- data types:
  `u16`, `s16`, `u32`, `s32`, `f16`, `f32`
- semantics:
  For each active lane, computes `%lhs mod %rhs` and writes the result to `%dst`.

### `pto.vdiv_fixed`

- covered ISA mnemonics:
  `VDIVF`
- syntax:
  `pto.vdiv_fixed %dst, %num, %den, %pg : !pto.vreg<Nxu16>, !pto.vreg<Nxu16>, !pto.vreg<Nxu16>, !pto.mask`
- operand roles:
  `%dst` is the destination vector, `%num` is the fixed-point numerator,
  `%den` is the divisor vector, and `%pg` is the zeroing predicate.
- data types:
  All vector operands are `u16`.
- semantics:
  For each active lane, computes `(%num << 16) / %den`, keeps the low 16 bits as
  a `0.16` fixed-point result, and saturates any quotient greater than `0xFFFF`
  to `0xFFFF`.

### `pto.vfma_dst`

- covered ISA mnemonics:
  `VMADD`, `VAXPY`
- syntax:
  `pto.vfma_dst %dst, %mul_lhs, %mul_rhs, %pg {scalar_rhs = true|false}`
- operand roles:
  `%dst` is both accumulator input and result, `%mul_lhs` is the vector source,
  `%mul_rhs` is the vector or scalar multiplier, and `%pg` is the governing
  predicate.
- data types:
  `VMADD`: `f16`, `bf16`, `f32`
  `VAXPY`: `f16`, `f32`
- semantics:
  `VMADD` computes `%dst = %mul_lhs * %dst + %mul_rhs` on active lanes, using
  the fused floating-point behavior of the ISA. `VAXPY` computes `%dst =
  %mul_rhs * %mul_lhs + %dst`, where `%mul_rhs` is a scalar shared-register
  coefficient broadcast across the active lanes.
- assertions and exceptions:
  `VMADD` is not interchangeable with a separate multiply followed by add;
  the fused rounding path is architecturally visible.

### `pto.vabs_and_diff`

- covered ISA mnemonics:
  `VABSDIF`, `VNEG`
- syntax:
  `pto.vabs_and_diff %dst, %lhs, %rhs?, %pg {mode = "absdiff|neg"}`
- operand roles:
  `%dst` is the destination vector, `%lhs` is the source vector, `%rhs` is the
  second source vector for `absdiff`, and `%pg` is the zeroing predicate.
- data types:
  `absdiff` accepts `u8`, `s8`, `u16`, `s16`, `u32`, `s32`, `f16`, or `f32`.
  `neg` preserves the source element type.
- semantics:
  `absdiff` computes `abs(lhs - rhs)` in full precision before the absolute
  operation and then truncates without implicit saturation. `neg` computes the
  lane-wise arithmetic negation of `%lhs` on the active lanes.

### `pto.vround_shift`

- covered ISA mnemonics:
  `VRND`, `VRNDS`, `VRNDI`, `VSHLI`, `VSHRI`
- syntax:
  `pto.vround_shift %dst, %src, %amount, %pg {mode = "round|shift_left|shift_right", amount_source = "vector|shared|imm", signed_shift = true|false}`
- operand roles:
  `%dst` is the destination vector, `%src` is the input vector, `%amount` is
  the per-lane, shared-register, or immediate shift amount, and `%pg` is the
  zeroing predicate.
- data types:
  round forms: `s16`, `s32`
  shift-left forms: `u8`, `u16`, `u32`
  shift-right forms: `b8`, `b16`, `b32` with `signed_shift` selecting logical
  versus arithmetic behavior
- semantics:
  Round forms compute `(src + (1 << (amount - 1))) >> amount`; if the shift
  magnitude exceeds the element width, the result is zero. Shift-left performs a
  logical left shift and yields zero when the shift magnitude exceeds the lane
  width. Shift-right performs logical or arithmetic right shift according to
  `signed_shift`; for arithmetic right shift, overlarge positive magnitudes
  yield `0` for non-negative inputs and `-1` for negative inputs.

### `pto.vavg`

- covered ISA mnemonics:
  `VAVG`
- syntax:
  `pto.vavg %dst, %lhs, %rhs, %pg {round = true|false}`
- operand roles:
  `%dst` is the destination vector, `%lhs` and `%rhs` are the averaged source
  vectors, and `%pg` is the zeroing predicate.
- data types:
  `u8`, `s8`, `u16`, `s16`
- semantics:
  Computes the lane-wise arithmetic average of `%lhs` and `%rhs` as
  `(lhs + rhs) >> 1`, optionally rounding toward positive infinity when the
  `round` bit is set.

### `pto.vtriad`

- covered ISA mnemonics:
  `VADD3`, `VADIF`, `VSAD`, `VSADDS`
- syntax:
  `pto.vtriad %dst, %lhs, %rhs, %pg {mode = "add3|adif|sad|sadds", scalar_rhs = true|false, saturate = true|false}`
- operand roles:
  `%dst` is the destination vector and, for `add3`, `adif`, and `sad`, also the
  accumulator input. `%lhs` and `%rhs` are the additional vector or scalar
  operands selected by the mode. `%pg` is the zeroing predicate.
- data types:
  `add3`, `adif`, and `sad` accept `u8`, `s8`, `u16`, `s16`, `u32`, or `s32`.
  `sadds` accepts `s16` or `s32`.
- semantics:
  `add3` computes `dst = dst + lhs + rhs`.
  `adif` computes `dst = lhs - rhs + dst`.
  `sad` computes `dst = abs(lhs - rhs) + dst`.
  `sadds` adds a scalar shared-register value to `%lhs` with signed saturation.

### `pto.vprelu`

- covered ISA mnemonics:
  `VPRELU`
- syntax:
  `pto.vprelu %dst, %src, %slope, %pg`
- operand roles:
  `%dst` is the destination vector, `%src` is the source activation vector,
  `%slope` is the negative-side multiplier vector, and `%pg` is the zeroing
  predicate.
- data types:
  `f16`, `f32`
- semantics:
  For each active lane, computes `%src` when `%src >= 0` and `%src * %slope`
  otherwise.

### `pto.vsat_addsub`

- covered ISA mnemonics:
  `VSADD`, `VSSUB`
- syntax:
  `pto.vsat_addsub %dst, %lhs, %rhs, %pg {mode = "add|sub"}`
- operand roles:
  `%dst` is the destination vector, `%lhs` and `%rhs` are the signed inputs, and
  `%pg` is the zeroing predicate.
- data types:
  `s16`, `s32`
- semantics:
  Computes lane-wise signed addition or subtraction and saturates the result to
  the destination lane range.

### `pto.vrsqrt`

- covered ISA mnemonics:
  `VRSQRT`
- syntax:
  `pto.vrsqrt %dst, %src, %pg`
- operand roles:
  `%dst` is the destination vector, `%src` is the source vector, and `%pg` is
  the zeroing predicate.
- data types:
  `f16`, `f32`
- semantics:
  For each active lane, computes `1 / sqrt(src)`.
- assertions and exceptions:
  A negative input raises an exception. `-0` yields `-inf`.

## 18. Reduction And Compression Families

ISA assertions for this family:

- Reduction families write their scalar or shortened-vector results into the
  low lanes of the destination vector and zero-fill the remaining lanes.
- Predicate operands in this chapter select which source elements participate in
  the reduction; they do not mask the writeback of the reduced result.

### `pto.vreduce_bound`

- covered ISA mnemonics:
  `VCBMAX`, `VCBMIN`
- syntax:
  `pto.vreduce_bound %dst, %match_pred, %src, %pg {mode = "max|min"}`
- operand roles:
  `%dst` holds the scalar reduction result in its lowest lane, `%match_pred`
  marks the source lanes equal to that result, `%src` is the reduced vector, and
  `%pg` selects the active lanes.
- data types:
  `u8`, `s8`, `u16`, `s16`, `u32`, `s32`, `f16`, `f32`
- semantics:
  Finds the global maximum or minimum over the active lanes, writes that value
  to the lowest lane of `%dst`, zero-fills the remaining lanes, and sets the
  corresponding bits in `%match_pred` for every source lane equal to the chosen
  bound.
- assertions and exceptions:
  For floating-point max, inactive lanes are treated as `-inf`; for floating
  min, they are treated as `+inf`. For integer max and min, inactive lanes are
  treated as the literal minimum or maximum value, respectively. If every lane
  is inactive, the sentinel value is written and `%match_pred` is all-zero.
  NaN propagates to the result and marks each NaN lane in `%match_pred`. For max
  the bound comparison treats `+0 > -0`; for min it treats `-0 < +0`. In the
  match predicate, both `+0` and `-0` lanes compare equal to the selected zero.

### `pto.vreduce_group`

- covered ISA mnemonics:
  `VCGADD`, `VCGMAX`, `VCGMIN`, `VCPADD`
- syntax:
  `pto.vreduce_group %dst, %src, %pg {mode = "add32B|max32B|min32B|pair_add"}`
- operand roles:
  `%dst` is the shortened-vector reduction result, `%src` is the source vector,
  and `%pg` selects the active input lanes.
- data types:
  `add32B`, `max32B`, and `min32B` accept `u8`, `s8`, `u16`, `s16`, `u32`,
  `s32`, `f16`, or `f32`. `pair_add` accepts `f16` or `f32`.
- semantics:
  `add32B` reduces all active elements within each 32-byte block and packs the
  block results contiguously into the low lanes of `%dst`. `max32B` and `min32B`
  compute the maximum or minimum of each block with the same block-wise packing.
  `pair_add` reduces every adjacent pair of lanes and writes the pairwise sums
  into the lower half of `%dst`.
- assertions and exceptions:
  Inactive lanes contribute `0` to additive reductions, `-inf` or the literal
  minimum to max reductions, and `+inf` or the literal maximum to min
  reductions. If all lanes of an additive reduction are inactive, the reduced
  value is `0` or `+0`.

### `pto.vsqueeze`

- covered ISA mnemonics:
  `VSQZ`, `VUSQZ`
- syntax:
  `pto.vsqueeze %dst, %src?, %pg {mode = "squeeze|unsqueeze", element_type = "b8|b16|b32", store_hint = true|false, sqzn = "SQZN"?}`
- operand roles:
  `%dst` is the squeezed or unsqueezed vector result, `%src` is the input vector
  for `squeeze`, `%pg` is the governing predicate, `store_hint` carries the
  ISA's `#st` hint, and `sqzn` names the SPR receiving the surviving-byte count.
- data types:
  `squeeze` and `unsqueeze` use `b8`, `b16`, or `b32` lane interpretation.
- semantics:
  `squeeze` compacts the active `%src` elements toward the least-significant end
  of the logical result sequence, writes the packed sequence into `%dst`, and
  zero-fills the remaining lanes. It writes the number of surviving bytes into
  `SQZN`. `unsqueeze` computes the prefix count of active predicate bits, with
  `dst[0] = 0` and `dst[i] = dst[i-1] + 1` when `pg[i-1]` is true.
- assertions and exceptions:
  When `store_hint` is set, the current `SQZN` value is queued for a later
  `VSTUR`; the total number of queued hints MUST match the total number of
  consuming stateful stores, and a `VSTUR` MUST separate consecutive squeezed
  results with `store_hint = true` to avoid hardware deadlock.

## 19. Wide Register Families

ISA assertions for this family:

- `W*` families operate on wide-register architectural state and MUST preserve
  the widening, sign-extension, and lane-selection rules encoded by the mnemonic
  suffix.
- The destination of a wide arithmetic op is not interchangeable with a normal
  vector register, even when the final packed result later returns to `vreg`.

### `pto.wdup`

- covered ISA mnemonics:
  `WDUPS`
- syntax:
  `pto.wdup %dst, %scalar {element_type = "u8|s8|u16|s16"}`
- operand roles:
  `%dst` is the destination wide register and `%scalar` is the shared-register
  source value.
- data types:
  The source scalar is interpreted as `u8`, `s8`, `u16`, or `s16` and widened
  accordingly.
- semantics:
  Broadcasts the source scalar into every active wide-register lane, zero- or
  sign-extending the source subword to the widened lane width required by the
  selected element type.

### `pto.wmov`

- covered ISA mnemonics:
  `WMOV`, `WMOVT`
- syntax:
  `pto.wmov %dst, %src, %pg? {element_type = "b8|b16", predicated = true|false}`
- operand roles:
  `%dst` is the destination wide register, `%src` is the source wide register,
  and `%pg` is the optional governing predicate for the merging form.
- data types:
  The lane interpretation is `b8` or `b16`.
- semantics:
  Copies wide-register lanes from `%src` to `%dst`. The predicated form updates
  only active lanes and uses merging behavior; the unpredicated form copies all
  lanes.

### `pto.wwide_add`

- covered ISA mnemonics:
  `WADD`, `WADDA`, `WSUB`, `WSUBA`, `WADDSUB`
- syntax:
  `pto.wwide_add %dst, %lhs, %rhs, %pg {mode = "add|add_acc|sub|sub_acc|addsub", element_type = "u8|s8|u16|s16"}`
- operand roles:
  `%dst` is either a pure result or an accumulator/result depending on `mode`,
  `%lhs` and `%rhs` are vector-register sources, and `%pg` is the zeroing
  predicate.
- data types:
  The source vector lanes are `u8`, `s8`, `u16`, or `s16`. The destination wide
  lanes are the corresponding widened signed or zero-extended integer lanes.
- semantics:
  `add` and `sub` compute lane-wise wide addition or subtraction and write the
  widened result to `%dst`. `add_acc` and `sub_acc` accumulate the widened sum
  or difference into the incoming `%dst` wide value. `addsub` computes `lhs +
  rhs - dst`. Signed element types sign-extend to the widened lane width and
  unsigned types zero-extend.

### `pto.wwide_mul`

- covered ISA mnemonics:
  `WMUL`, `WMULA`, `WMULS`, `WMULAS`, `WMULM`
- syntax:
  `pto.wwide_mul %dst, %lhs, %rhs, %pg? {mode = "mul|mula|muls|mulas|mulm", lhs_type = "...", rhs_type = "...", rhs_source = "vector|shared|vector_pair", part = "even|odd"?}`
- operand roles:
  `%dst` is the wide result or accumulator/result, `%lhs` is the vector source,
  `%rhs` is a vector, shared-register, or paired-vector source depending on the
  mode, `%pg` is the governing predicate when present, and `part` selects even
  or odd source-element participation for the forms that consume only half of a
  wider source lane stream.
- data types:
  The allowed source pairings are exactly those encoded by the ISA suffix:
  `u8s8`, `u8u8`, `u8s16`, `s8s8`, `s8s16`, `u16u16`, `u16s32`, `u16s16`,
  `u16u32`, `s16s16`, `s16s32`, `s16u32`, and the mixed `u8s16` / `s8s16`
  `WMULM` forms.
- semantics:
  Computes the ISA-defined wide multiplication, multiply-accumulate, or
  mixed-source multiply variant. The result lane width depends on the operand
  pairing: 8-bit by 8-bit and 8-bit by 16-bit families widen to 24-bit lanes,
  while 16-bit by 16-bit and 16-bit by 32-bit families widen to 48-bit lanes.
  Mixed-pair forms that read `{Vm, Vm+1}` or a shared-register scalar MUST
  preserve the exact source-lane pairing required by the mnemonic suffix.
- assertions and exceptions:
  For `u8s16` / `s8s16` mixed forms, `Vm` denotes the even-numbered base of the
  `{Vm, Vm+1}` pair. Forms with `#part` consume only even or only odd elements
  as specified by that selector.

### `pto.wpack`

- covered ISA mnemonics:
  `WPACK`, `WPACKT`, `WPACKA`, `WPACKI`, `WPACKIT`, `WPACKIA`, `WPACKS`,
  `WPACKST`, `WPACKSA`
- syntax:
  `pto.wpack %dst, %src, %shift, %pg? {shift_source = "vector|shared|imm", transpose = true|false, accum_part = "none|even|odd", round_sat = true|false, result_type = "..."}`
- operand roles:
  `%dst` is the destination vector register, `%src` is the wide-register source,
  `%shift` is the per-lane vector shift, shared-register shift, or immediate
  shift amount, `%pg` is the zeroing predicate for predicated forms, and
  `accum_part` selects even or odd packing when the source wide register carries
  paired substreams.
- data types:
  The destination type is one of the ISA suffix-defined results such as
  `s242u8`, `s242s8`, `s482u16`, `s482s16`, `s242u16`, `s242s16`, `s482s32`,
  or `s642s32`.
- semantics:
  Right-shifts each wide-register lane by the selected shift amount, optionally
  adds the last shifted-off bit for rounding, optionally saturates the result to
  the destination lane type, and writes the truncated destination-width value
  into `%dst`. When the shift amount exceeds 32, the shift is clamped to 32.
  `result_type` preserves the mnemonic suffix such as `s242u8`, `s482u16`, or
  `s642s32`.

### `pto.wcvt48`

- covered ISA mnemonics:
  `WCVT48`
- syntax:
  `pto.wcvt48 %dst, %hi32, %lo16, {part = "even|odd"}`
- operand roles:
  `%dst` is the destination wide register, `%hi32` supplies the high 32 bits of
  each wide lane, and `%lo16` supplies the low 16-bit slice taken from even or
  odd vector lanes.
- data types:
  `%hi32` is a 32-bit vector source and `%lo16` is a 16-bit vector source. The
  destination wide lanes are 48-bit integers.
- semantics:
  Concatenates 32 bits from each lane of `%hi32` with 16 bits from the even or
  odd lanes of `%lo16`, producing 48-bit wide-register lanes written to the
  selected even or odd half of `%dst`.

## 20. Vision And Histogram Families

ISA assertions for this family:

- These families are not generic arithmetic shorthands; each one preserves its
  ISA-defined window shape, configuration word layout, and block accumulation
  rule.
- Memory-address operands in this section refer to architecturally aligned
  buffer addresses rather than arbitrary byte pointers.

### `pto.vfifr1`

- covered ISA mnemonics:
  `FIFR1`, `FMAX`, `FMIN`
- syntax:
  `pto.vfifr1 %dst_addr, %src_addr, %cfg_xm, %cfg_xt? {mode = "filter|max|min", input_type = "u8|s8|u16|s16", round = true|false}`
- operand roles:
  `%dst_addr` is the destination address, `%src_addr` is the source address,
  `%cfg_xm` is the configuration register containing the window size, strides,
  repeat count, and first-row coefficients, and `%cfg_xt` carries the row
  multiplication factors for filter mode.
- data types:
  `filter` accepts `u8`, `s8`, `u16`, or `s16` inputs and produces `s16`
  outputs. `max` and `min` accept `u8` inputs and produce `u8` outputs.
- semantics:
  Executes the fast-image-filter pipeline directly in the vector sequencer.
  `filter` computes the separable 3-tap or 5-tap FIR described by the
  coefficient matrix. `max` and `min` compute the corresponding morphological
  filter over the same streaming window. `cfg_xm` also carries source stride,
  destination stride, and input-line count, and the line count MUST be at least
  the selected window size.
- assertions and exceptions:
  Source and destination addresses MUST be 32-byte aligned. Filter coefficients
  are signed 5-bit integers. If rounding is enabled, both the intermediate and
  final right shifts round by adding the last shifted-off bit and then saturate
  on overflow.

### `pto.wfifr2`

- covered ISA mnemonics:
  `WFIFR2`, `WFIFR2A`, `WFIFR2S`
- syntax:
  `pto.wfifr2 %dst, %src, %coeffs {mode = "fir|accum|sub", input_type = "u8|s8|u16|s16"}`
- operand roles:
  `%dst` is the destination wide register, `%src` is the even-numbered source
  vector-register pair base, and `%coeffs` is the shared-register coefficient
  pack holding `C1..C5`.
- data types:
  `%src` uses `u8`, `s8`, `u16`, or `s16` input lanes. `%coeffs` contains five
  signed 5-bit coefficients packed into one shared register.
- semantics:
  Reads `VL_16 + 4` source elements from `%src` and `%src + 1`, computes the
  5-tap one-dimensional FIR defined by the coefficient pack, and writes the
  result into `%dst`. `accum` adds the FIR result into the incoming wide value;
  `sub` subtracts the FIR result from the incoming wide value.

### `pto.vintegral`

- covered ISA mnemonics:
  `VINTEGRALv2`
- syntax:
  `pto.vintegral %dst_even, %dst_odd, %src : !pto.vreg<Nxu16>, !pto.vreg<Nxu16>, !pto.vreg<Mxu8>`
- operand roles:
  `%dst_even` receives the lower half of the prefix sums, `%dst_odd` receives
  the upper half, and `%src` is the input vector of `u8` elements.
- data types:
  `%src` is `u8`; both results are `u16`.
- semantics:
  Computes the integral-prefix sums of the input `u8` lanes and writes the lower
  half of the `u16` prefix results to `%dst_even` and the upper half to
  `%dst_odd`. `%dst_even` must name an even register pair base in the ISA model.
- assertions and exceptions:
  The second result is incomplete until the program adds `%dst_even` into
  `%dst_odd`; the final integral vector is `{dst_odd + dst_even, dst_even}`.
  For vector lengths at or below 256 bytes, this family does not overflow. For
  longer vectors, overflow truncates to 16 bits before writeback.

### `pto.vhistogram`

- covered ISA mnemonics:
  `DHISTv2`, `CHISTv2`
- syntax:
  `pto.vhistogram %dst, %src, %pg, #bin {mode = "distribution|cumulative"}`
- operand roles:
  `%dst` is the destination vector of `u16` bin accumulators, `%src` is the
  source vector of `u8` samples, `%pg` selects active input elements, and `#bin`
  chooses which contiguous range of the 256 total histogram bins is represented
  by the destination vector.
- data types:
  `%src` is `u8`, `%dst` is `u16`, and `#bin` is a 4-bit unsigned immediate.
- semantics:
  `distribution` accumulates the per-bin counts for the selected bin range.
  `cumulative` accumulates the cumulative histogram for the same range. `%pg`
  filters which input elements contribute, but it does not mask destination
  writes: every destination bin accumulator is updated.
- assertions and exceptions:
  `#bin` MUST be less than `256 / VL_16` for the active vector length.

## 21. Detection And Suppression Families

ISA assertions for this family:

- These families are specialized state-transform instructions. Their
  destination state is defined by the matrix, predicate, and SPR layout rules
  below and MUST NOT be rewritten as generic reduction ops.

### `pto.prpnset`

- covered ISA mnemonics:
  `PRPNSET`
- syntax:
  `pto.prpnset %index, %pg, %rpn_cor_ir {element_type = "f16"}`
- operand roles:
  `%index` is the shared-register selector for the slice of `RPN_COR_IR`,
  `%pg` is the source predicate, and `%rpn_cor_ir` is the special-register state
  being updated.
- data types:
  The defined contract is `f16` only. `%index` is an integer shared-register
  selector and `%pg` is the corresponding predicate input.
- semantics:
  For `f16`, reduces every 16-element block of `%pg` with OR, selects the slice
  of `RPN_COR_IR` indexed by `%index`, and ORs each block result into that slice
  in place.
- assertions and exceptions:
  The currently defined ISA contract covers `f16` only; any other type is
  outside this specification.

### `pto.rpn_cor_diag2`

- covered ISA mnemonics:
  `RPN_COR_DIAG2`
- syntax:
  `pto.rpn_cor_diag2 %dst_addr, %src_addr, %rpn_cor_ir {element_type = "f16"}`
- operand roles:
  `%dst_addr` is the destination address for the suppression vector,
  `%src_addr` is the source address of the suppression matrix, and
  `%rpn_cor_ir` is the incoming diagonal suppression-state SPR.
- data types:
  The defined contract is `f16` only. The destination is a 16-element
  suppression vector of 16-bit results; the source is a 16x16 matrix of 2-bit
  suppression entries.
- semantics:
  Reads the 16x16x2-bit suppression matrix from `%src_addr`, computes the 16
  suppression-vector bits using the staged diagonal recurrence with
  `RPN_COR_IR`, and writes the 16 packed 16-bit suppression outputs to
  `%dst_addr`.
- assertions and exceptions:
  Source and destination addresses MUST be 32-byte aligned. The currently
  defined ISA contract covers `f16` only.
