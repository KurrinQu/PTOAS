# Phase 2: PTO Lowering - Research

**Researched:** 2026-03-18
**Domain:** PTO-to-A5VM lowering for the `Abs` path with PTO semantic preservation
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Implementation Decisions

### TLOAD semantic preservation
- Phase 2 should use a template-mirroring lowering contract for `TLOAD`, not an `Abs`-only shortcut.
- `TLOAD` lowering should carry essentially all PTO-op decision inputs that matter to template dispatch, not only the smallest subset used by the current sample.
- `pad_mode`, `pad_value`, `left_padding_num`, `right_padding_num`, `init_out_buffer`, and `init_condition` should all be acknowledged in the lowering contract even if the `Abs` sample does not actively exercise them yet.
- The lowering should preserve PTO view-to-tile mapping traces rather than collapsing directly to only the final codegen-facing valid region.
- Layout rules, valid-row/valid-col information, padding/init behavior, and source/domain information are all important and should not be casually dropped.

### TSTORE branch structure
- The Phase 2 lowering contract and code skeleton should explicitly show `ACC`, `VEC`, and `MAT` source-tile branches, even though the current `Abs` path only needs the `VEC` branch implemented strictly.
- PTO-side layout rules that influence `TSTORE` selection should be part of the lowering input rather than inferred implicitly later.
- For the current slice, the `VEC` branch should be implemented strictly; `ACC` and `MAT` branches may lower to explicit TODO or placeholder paths, but they should exist in the entrypoint and keep their decision inputs visible.
- Source tile domain, destination layout/shape/stride, and valid row/column are all equally important selection inputs for `TSTORE` lowering.

### TABS alignment standard
- `lowerTABS` should align as closely as practical to the PTO template and implementation decision structure, not only to the observable `abs` result.
- PTO-side restrictions such as dtype/domain/valid-shape compatibility should be enforced in lowering pre-checks before building `a5vm`.
- `TABS` should become the standard lowering template for future unary ops such as `TNEG` and `TLOG`, not a one-off special case.
- Type restrictions, valid-region handling, and input/output tile relationship consistency are all critical and must not drift from PTO behavior.

### Reusable framework boundary
- Phase 2 should strike a balance: preserve `Abs` path precision while shaping the lowering entrypoints and contracts for future expansion.
- The minimum acceptable extensibility target is that both a future unary op and a future load/store variant can be added without rewriting the lowering architecture.
- Lowering should use three strong PTO entrypoints (`lowerTLOAD`, `lowerTABS`, `lowerTSTORE`) backed by shared helper utilities.
- Important PTO decision logic should remain visible at each PTO entrypoint; repeated mechanics and repeated branch detail can move into shared helpers.
- Future extensibility is prioritized over local short-term simplification, but not at the cost of making the `Abs` path inaccurate.
- The code should make PTO template-to-lowering correspondence readable when someone inspects it later.
- For current non-`Abs` branches such as `ACC`/`MAT`, the interface entrypoints should contain explicit TODO branches so future completion points are obvious.

### Claude's Discretion
- Exact helper names and file split for shared lowering utilities
- Exact representation of PTO view-to-tile mapping traces inside the lowering contract
- Exact placeholder form for non-`Abs` `ACC`/`MAT` branches so long as the branch structure and inputs remain visible

### Deferred Ideas (OUT OF SCOPE)
## Deferred Ideas

- Broader PTO op lowering beyond the `Abs` path remains outside this phase, even though the framework should be designed to support it later.
- Full implementation of `ACC` and `MAT` `TSTORE` branches is deferred; Phase 2 should make the branches explicit and preserve their inputs, but not necessarily complete them for execution.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PTO-01 | Developer can lower PTO `TLOAD` on the `Abs` path into `a5vm` operations while preserving the PTO-side layout, shape, and valid-region decisions needed for backend code selection. | Use a `lowerTLOAD` entrypoint that extracts source layout, source shape/stride, destination tile domain/layout, valid row/col, padding/init operands, and partition-view mapping trace into an explicit lowering contract instead of emitting a sample-specific `a5vm.load`. |
| PTO-02 | Developer can lower PTO `TABS` on the `Abs` path into `a5vm` operations in a way that matches existing PTO parameter behavior and unary-op template dispatch intent. | Model `lowerTABS` as a reusable unary lowering helper with pre-checks for vec-domain, row-major layout, supported dtype, and src/dst valid-shape equality before creating `a5vm.abs`. |
| PTO-03 | Developer can lower PTO `TSTORE` on the `Abs` path into `a5vm` operations while preserving the PTO-side source tile domain and destination layout behavior needed for code selection. | Build `lowerTSTORE` around explicit `VEC` / `ACC` / `MAT` branches, preserving source tile domain plus destination layout, shape, stride, and valid-region data; implement `VEC` now and keep `ACC` / `MAT` as explicit placeholders. |
| PTO-04 | Developer can add new PTO-to-A5VM lowerings through the same framework without changing the backend architecture established for `Abs`. | Keep three visible PTO entrypoints with shared extraction/validation helpers and reusable unary/load-store contracts so later ops add cases rather than replace architecture. |
</phase_requirements>

## Summary

Phase 2 should be planned as a semantic-lowering layer between PTO ops and the Phase 1 `a5vm` dialect, not as another direct codegen rewrite like [`PTOToEmitC.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToEmitC.cpp). The current backend flattens `pto.tload`, `pto.tabs`, and `pto.tstore` to opaque `TLOAD`/`TABS`/`TSTORE` calls with almost none of the template-dispatch inputs preserved. That is precisely the behavior this phase must replace.

The authoritative behavior lives in the PTO IR surface and the CANN PTO template implementations. `TLoadOp` already carries padding/init operands in ODS, `TStoreOp` already exposes tile-domain differences via pipe behavior and source types, and `TAbsOp` already has verifier checks for compatible element types and shapes. The A2/A3 PTO headers make the hidden selection criteria explicit: `TLOAD_IMPL` depends on source layout, source shape/stride, destination tile kind/layout, and destination valid region; `TSTORE_IMPL` branches first on source tile domain (`Vec`, `Acc`, `Mat`) and then on destination layout/shape/stride and quantization variants; `TABS_IMPL` is the generic unary template shape with dtype, vec-domain, row-major, and valid-shape checks.

**Primary recommendation:** Implement a dedicated PTO-to-A5VM lowering pass with three visible entrypoints, shared semantic-extraction helpers, and explicit preservation of template-dispatch inputs as typed `a5vm` metadata, while keeping non-`Abs` `ACC`/`MAT` store paths as explicit TODO branches rather than hiding them.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| LLVM | 19.1.7 | Pass infrastructure, diagnostics, type support | Already pinned by the workspace toolchain and matched by MLIR. |
| MLIR | 19.1.7 | Conversion pass, pattern rewriting, dialect interop | The repo already uses MLIR passes and dialect conversion for PTO transforms. |
| PTO IR / PTO Transforms | workspace | Source PTO op surface and transform entrypoints | Phase 2 must lower from existing PTO ops instead of inventing a parallel frontend contract. |
| Ascend CANN PTO headers | 8.5.0 | Official semantic reference for `TLOAD`, `TABS`, `TSTORE` template behavior | These headers define the real dispatch inputs and restrictions that lowering must preserve. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Phase 1 `a5vm` dialect | planned in workspace | Target IR for hardware-facing lowering | Use once Phase 1 lands; Phase 2 should depend on its op/type surface, not recreate it. |
| MLIR `RewritePattern` / `ConversionPattern` | 19.1.7 | PTO-op lowering implementation | Use for op-by-op lowering while keeping shared dialects intact. |
| `FileCheck` with source `RUN:` lines | LLVM 19.1.7 toolchain | Fast structural verification of lowered IR | Use for phase fixtures because the repo already relies on `RUN:` + `FileCheck` tests. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Dedicated PTO-to-A5VM pass | Extend `PTOToEmitC.cpp` with `a5vm`-specific branches | Faster short-term, but it keeps the wrong abstraction boundary and hides PTO semantic contracts. |
| Explicit lowering contracts per PTO family | Infer backend details inside the A5VM emitter | Simpler locally, but it delays validation and loses traceability from PTO template decisions. |
| Shared helper library under `lib/PTO/Transforms/` | One monolithic `matchAndRewrite` body per op | Works for `Abs`, but it does not meet PTO-04 extensibility. |

**Toolchain verification:**
```bash
sed -n '1,20p' /data/mouliangyu/projects/github.com/llvm/llvm-project/install/lib/cmake/llvm/LLVMConfigVersion.cmake
sed -n '1,20p' /data/mouliangyu/projects/github.com/llvm/llvm-project/install/lib/cmake/mlir/MLIRConfigVersion.cmake
```

**Verified local versions:**
- LLVM `19.1.7`
- MLIR `19.1.7`
- CANN PTO reference tree `8.5.0` at `/usr/local/Ascend/cann-8.5.0/...`

## Architecture Patterns

### Recommended Project Structure
```text
include/PTO/Transforms/
├── Passes.h                  # new pass declaration for PTO->A5VM lowering
└── PTOToA5VM.h               # optional shared lowering contract declarations

lib/PTO/Transforms/
├── PTOToA5VM.cpp             # pass entrypoint and pattern population
├── PTOToA5VMLowering.h       # optional internal structs/helpers
└── PTOToA5VMLowering.cpp     # semantic extraction + per-op lowering helpers
```

### Pattern 1: Strong PTO Entrypoints, Shared Mechanics
**What:** Keep `lowerTLOAD`, `lowerTABS`, and `lowerTSTORE` as visible helper entrypoints, with shared utilities only for repeated extraction and verification.
**When to use:** For all Phase 2 lowering code.
**Example:**
```c++
static FailureOr<LoadContract> lowerTLOAD(pto::TLoadOp op,
                                          PatternRewriter &rewriter);
static FailureOr<UnaryContract> lowerTABS(pto::TAbsOp op,
                                          PatternRewriter &rewriter);
static FailureOr<StoreContract> lowerTSTORE(pto::TStoreOp op,
                                            PatternRewriter &rewriter);
```
Source: project context in [`02-CONTEXT.md`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/.planning/phases/02-pto-lowering/02-CONTEXT.md) plus existing transform organization in [`Passes.h`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/Transforms/Passes.h)

### Pattern 2: Extract a Lowering Contract Before Building `a5vm`
**What:** Separate PTO semantic extraction from `a5vm` op creation. First build a small struct containing the decisions the PTO template would dispatch on, then lower that struct into `a5vm`.
**When to use:** For `TLOAD`, `TSTORE`, and reusable unary lowering.
**Example:**
```c++
struct LoadContract {
  Value srcBase;
  Value dstTile;
  pto::Layout srcLayout;
  SmallVector<Value> srcShape;
  SmallVector<Value> srcStride;
  Value validRow;
  Value validCol;
  Value padValue;
  Value leftPadding;
  Value rightPadding;
  bool initOutBuffer;
};
```
Source: PTO op surface in [`PTOOps.td`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/IR/PTOOps.td) and official PTO template parameters in `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TLoad.hpp`

### Pattern 3: Mirror PTO Template Branching at the Lowering Boundary
**What:** The lowering helper should expose the same first-order branch points as the PTO template implementation.
**When to use:** Especially for `TSTORE`, and secondarily for unary-family helpers.
**Example:**
```c++
switch (storeContract.srcDomain) {
case TileDomain::Vec:
  return lowerVecStore(storeContract, rewriter);
case TileDomain::Acc:
  return emitExplicitTodo(op, "ACC TSTORE lowering not implemented yet");
case TileDomain::Mat:
  return emitExplicitTodo(op, "MAT TSTORE lowering not implemented yet");
}
```
Source: official branch structure in `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TStore.hpp`

### Pattern 4: Reusable Unary Lowering Template
**What:** Make `TABS` the first instance of a general unary-op lowering helper that performs common checks, then selects the `a5vm` op family.
**When to use:** For `TABS` now, `TNEG` / `TLOG` later.
**Example:**
```c++
template <typename PTOUnaryOp, typename BuildFn>
static LogicalResult lowerVecUnaryOp(PTOUnaryOp op, PatternRewriter &rewriter,
                                     BuildFn &&buildA5VMOp);
```
Source: reusable unary pattern in `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TUnaryOp.hpp` and analogous reusable binary structure in `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TAdd.hpp`

### Anti-Patterns to Avoid
- **Direct opaque-call replacement:** Reproducing the current `emitc::CallOpaqueOp("TLOAD")` / `("TSTORE")` / `("TABS")` behavior in a different file still loses semantics.
- **Emitter-side semantic recovery:** If Phase 3 has to reconstruct tile domain, layout, or valid shape by inspecting old PTO values, Phase 2 failed.
- **Abs-only contracts:** Hardcoding only the specific `32x32` vec case from `test/samples/Abs/abs.py` violates locked extensibility requirements.
- **Hidden unsupported branches:** `ACC` and `MAT` paths must stay visible as placeholders, not disappear into a default error.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PTO semantic truth | A new ad hoc semantic model separate from PTO IR and CANN headers | PTO ODS/verifiers plus CANN PTO template behavior as the lowering contract source | The IR already carries much of the surface, and the headers define the missing dispatch rules. |
| Per-op one-off logic | Three unrelated rewrite bodies with duplicated shape/layout extraction | Shared contract extractors and validators | Keeps `Abs` accurate while making future ops additive. |
| Unary-op semantics | A one-off `TABS` helper | Generic vec-unary lowering template | `TABS`, `TNEG`, and `TLOG` share the same core restrictions and valid-shape handling pattern. |
| Domain detection | Stringly-typed attrs guessed late in emission | Source tile type / memory-space driven domain classification in lowering | `TSTORE` pipe/domain behavior is already encoded in PTO types and verifier assumptions. |

**Key insight:** The dangerous complexity in this phase is not IR rewriting itself; it is preserving exactly the semantic decisions that PTO template dispatch depends on. The plan should therefore spend effort on contract extraction and validation, not on clever pattern-matching shortcuts.

## Common Pitfalls

### Pitfall 1: Treating `TLOAD` as Only “address + valid shape”
**What goes wrong:** Lowering preserves only the final vec load shape and drops padding/init inputs and source layout/stride context.
**Why it happens:** The `Abs` sample is simple and the current backend already ignores most operands.
**How to avoid:** Keep all `TLoadOp` optional operands and source layout/shape/stride information in the lowering contract, even if some are not yet consumed by the current `Abs` lowering path.
**Warning signs:** `lowerTLOAD` reads only `src`, `dst`, and maybe `valid_row`/`valid_col`.

### Pitfall 2: Collapsing Partition Views Too Early
**What goes wrong:** The lowering erases PTO view-to-tile mapping trace and stores only flattened codegen metadata.
**Why it happens:** It is tempting to normalize everything immediately once a memref base is available.
**How to avoid:** Preserve partition offsets/sizes or an equivalent trace object in the lowering contract until `a5vm` metadata has been created.
**Warning signs:** Planner proposes deriving only “final region” values with no record of PTO view slicing.

### Pitfall 3: Hiding `TSTORE` Domain Branching
**What goes wrong:** `TSTORE` becomes a single generic helper with a default path, and `ACC`/`MAT` distinctions disappear.
**Why it happens:** Only the `VEC` path is required for `Abs`, so unsupported paths are ignored.
**How to avoid:** Make `srcDomain` an explicit lowering discriminator and keep TODO branches for unsupported domains in the entrypoint.
**Warning signs:** No visible `ACC` or `MAT` branch in the new pass.

### Pitfall 4: Using `TABS` as a Special Case Instead of a Template
**What goes wrong:** `TABS` lowering hardcodes abs-specific checks and shape logic that must later be rewritten for `TNEG` or `TLOG`.
**Why it happens:** `Abs` is the only unary op in scope today.
**How to avoid:** Factor common vec-unary validation now: supported dtype set, vec-domain restriction, row-major requirement, and src/dst valid-shape equality.
**Warning signs:** Helper names and contracts mention only `Abs` rather than unary behavior.

### Pitfall 5: Confusing Current PTO Verifiers With Full Backend Readiness
**What goes wrong:** Planning assumes existing PTO op verifiers are enough and skips lowering-time pre-checks.
**Why it happens:** `TAbsOp::verify()` already checks some compatibility.
**How to avoid:** Add lowering-time checks for backend-relevant restrictions that are explicit in the PTO templates but not fully encoded in ODS/verifiers today.
**Warning signs:** Planner says “verifier already guarantees it” without citing the A2/A3 template constraints.

### Pitfall 6: Planning Phase 2 as Executable Before Phase 1 Lands
**What goes wrong:** Tasks assume `a5vm` ops, pass declarations, and backend flags already exist even though the repo does not yet contain them.
**Why it happens:** Phase documents were created before implementation.
**How to avoid:** Treat Phase 1 `a5vm` op/type/pass surfaces as prerequisites or sequence the work so Phase 2 begins only once those artifacts exist.
**Warning signs:** Tasks reference `createLowerPTOToA5VMPass` or `--pto-backend=a5vm` as if they already compile.

## Code Examples

Verified patterns from official and repo-local sources:

### Current `TLOAD` Flattening That Phase 2 Must Replace
```c++
rewriter.create<emitc::CallOpaqueOp>(
    op.getLoc(), TypeRange{}, "TLOAD",
    ArrayAttr{}, ArrayAttr{},
    ValueRange{dst, srcArg});
```
Source: [`PTOToEmitC.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToEmitC.cpp)

### PTO `TLoadOp` Already Exposes The Lost Operands
```tablegen
let arguments = (ins
  PTODpsType:$src,
  PTODpsType:$dst,
  OptionalAttr<PTO_PadModeAttr>:$pad_mode,
  Optional<AnyType>:$pad_value,
  Optional<Index>:$left_padding_num,
  Optional<AnyType>:$right_padding_num,
  DefaultValuedOptionalAttr<BoolAttr, "false">:$init_out_buffer,
  Optional<AnyType>:$init_condition
);
```
Source: [`PTOOps.td`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/IR/PTOOps.td)

### Official `TABS_IMPL` Restriction Pattern
```c++
static_assert(TileData::Loc == TileType::Vec, "TABS: TileType of src and dst tiles must be TileType::Vec.");
static_assert(TileData::isRowMajor, "TABS: Not supported Layout type");
PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "TABS: Number of columns of src and dst must be the same.");
PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "TABS: Number of rows of src and dst must be the same.");
```
Source: `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TUnaryOp.hpp`

### Official `TSTORE_IMPL` Domain Split
```c++
if constexpr (TileData::Loc == pto::TileType::Vec) {
  ...
} else if constexpr (TileData::Loc == pto::TileType::Acc) {
  ...
} else if constexpr (TileData::Loc == pto::TileType::Mat) {
  ...
}
```
Source: `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TStore.hpp`

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Direct lowering to opaque EmitC calls | Preserve PTO semantics through a first-class `a5vm` lowering layer | Locked by project planning on 2026-03-18 | Phase 2 should not add more opaque-call lowering; it should move semantic decisions into `a5vm` metadata/contracts. |
| Sample-specific unary lowering | Reusable unary lowering template rooted in PTO template rules | Locked by Phase 2 context on 2026-03-18 | `TABS` should establish the standard unary shape for later ops. |
| Implicit future store expansion | Explicit visible `VEC` / `ACC` / `MAT` lowering branches | Locked by Phase 2 context on 2026-03-18 | Planner should reserve structure for unsupported paths now rather than retrofit it later. |

**Deprecated/outdated:**
- Continuing to use [`PTOToEmitC.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToEmitC.cpp) as the semantic reference for `TLOAD` / `TABS` / `TSTORE`: it is useful only as the replacement boundary, not as the desired behavior.

## Open Questions

1. **What exact `a5vm` op/attribute surface from Phase 1 will Phase 2 target?**
   - What we know: Phase 1 research expects minimal `a5vm.load`, `a5vm.abs`, and `a5vm.store` ops plus typed metadata.
   - What's unclear: The repo does not yet contain those ops, pass factories, or backend flags.
   - Recommendation: Treat finalized Phase 1 op/attr names and pass entrypoints as a planning prerequisite for Phase 2 implementation.

2. **How should PTO view-to-tile mapping trace be represented in `a5vm` metadata?**
   - What we know: The user explicitly wants that trace preserved, not flattened away.
   - What's unclear: Whether the best representation is offsets/sizes attrs, a custom trace attr, or structured helper attrs on load/store ops.
   - Recommendation: Pick the smallest typed representation that preserves partition offsets/sizes and derived valid-region facts without embedding PTO ops directly in `a5vm`.

3. **Should unsupported `ACC` / `MAT` store branches fail the pass or emit placeholders?**
   - What we know: The user wants explicit TODO branches and visible inputs.
   - What's unclear: Whether current backend policy prefers hard failure, preserved placeholder ops, or diagnostic comments for unsupported branches.
   - Recommendation: Plan for explicit diagnostics that fail lowering by default unless Phase 1 placeholder policy already established a sanctioned unresolved-op mechanism.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | MLIR source tests with `RUN:` + `FileCheck`, plus `ctest` smoke checks |
| Config file | none committed in source tree; current lit config appears external or build-generated |
| Quick run command | `./build/tools/ptoas/ptoas test/phase2/<case>.mlir -o - | FileCheck test/phase2/<case>.mlir` |
| Full suite command | `ctest --test-dir build --output-on-failure` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PTO-01 | `pto.tload` lowering preserves layout/shape/valid-region/padding inputs into `a5vm` metadata on the `Abs` path | integration | `./build/tools/ptoas/ptoas test/phase2/pto_tload_abs_lowering.mlir -o - | FileCheck test/phase2/pto_tload_abs_lowering.mlir` | ❌ Wave 0 |
| PTO-02 | `pto.tabs` lowering enforces unary restrictions and emits `a5vm.abs` with matched semantics | integration | `./build/tools/ptoas/ptoas test/phase2/pto_tabs_abs_lowering.mlir -o - | FileCheck test/phase2/pto_tabs_abs_lowering.mlir` | ❌ Wave 0 |
| PTO-03 | `pto.tstore` lowering preserves source tile domain and destination layout behavior for the `Abs` vec path | integration | `./build/tools/ptoas/ptoas test/phase2/pto_tstore_abs_lowering.mlir -o - | FileCheck test/phase2/pto_tstore_abs_lowering.mlir` | ❌ Wave 0 |
| PTO-04 | Shared lowering framework supports a second unary or store variant without architectural rewrite | unit/integration | `./build/tools/ptoas/ptoas test/phase2/pto_lowering_framework_reuse.mlir -o - | FileCheck test/phase2/pto_lowering_framework_reuse.mlir` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `./build/tools/ptoas/ptoas test/phase2/<case>.mlir -o - | FileCheck test/phase2/<case>.mlir`
- **Per wave merge:** `ctest --test-dir build --output-on-failure`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `test/phase2/pto_tload_abs_lowering.mlir` — covers PTO-01 with positive checks for preserved metadata and negative checks for missing required inputs
- [ ] `test/phase2/pto_tabs_abs_lowering.mlir` — covers PTO-02, including illegal dtype/layout/domain cases
- [ ] `test/phase2/pto_tstore_abs_lowering.mlir` — covers PTO-03 vec-path lowering and explicit unsupported-branch diagnostics
- [ ] `test/phase2/pto_lowering_framework_reuse.mlir` — covers PTO-04 by exercising shared helper structure beyond a single hardcoded op body
- [ ] A documented invocation path for `test/phase2/*.mlir` fixtures if no committed lit config is introduced during Phase 1

## Sources

### Primary (HIGH confidence)
- [`include/PTO/IR/PTOOps.td`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/IR/PTOOps.td) - PTO op surfaces for `TLoadOp`, `TStoreOp`, `TAbsOp`
- [`lib/PTO/IR/PTO.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/IR/PTO.cpp) - current verifier behavior for `TLoadOp`, `TStoreOp`, `TAbsOp`
- [`lib/PTO/Transforms/PTOToEmitC.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToEmitC.cpp) - current backend lowering boundary that Phase 2 replaces
- [`tools/ptoas/ptoas.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/tools/ptoas/ptoas.cpp) - current pipeline/emission integration surface
- [`test/samples/Abs/abs.py`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/test/samples/Abs/abs.py) - concrete `Abs` path exercised by v1 scope
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/common/pto_instr.hpp` - public PTO instruction entrypoints
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TLoad.hpp` - `TLOAD_IMPL` constraints and layout/shape/stride/valid-region inputs
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TStore.hpp` - `TSTORE_IMPL` branch structure and per-domain constraints
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TUnaryOp.hpp` - `TABS_IMPL` unary restriction pattern
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TAdd.hpp` - reusable template pattern for future op-family shaping
- [`01-RESEARCH.md`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/.planning/phases/01-a5vm-foundation/01-RESEARCH.md) - locked Phase 1 backend boundary and toolchain context

### Secondary (MEDIUM confidence)
- [`01-VALIDATION.md`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/.planning/phases/01-a5vm-foundation/01-VALIDATION.md) - prior validation style and expected Wave 0 structure
- [`ROADMAP.md`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/.planning/ROADMAP.md) - phase sequencing and success criteria

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - derived from repo-local toolchain files, existing code, and official local PTO headers
- Architecture: HIGH - derived from locked context decisions plus current repo transform structure and official PTO template branch patterns
- Pitfalls: HIGH - based on direct gaps between current `EmitC` lowering and required PTO semantic preservation

**Research date:** 2026-03-18
**Valid until:** 2026-04-17
