# Phase 2: PTO Lowering - Context

**Gathered:** 2026-03-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement PTO-to-A5VM lowering helpers that preserve existing PTO-side semantic decisions for the `Abs` path. This phase is about the lowering contract and lowering entrypoints for `TLOAD`, `TABS`, and `TSTORE`, with behavior aligned to PTO parameter semantics and template decision structure. It does not expand scope to unrelated PTO operations, but it should leave a lowering framework that can grow without being redesigned.

</domain>

<decisions>
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

</decisions>

<specifics>
## Specific Ideas

- The user wants `lowerTLOAD`, `lowerTABS`, and `lowerTSTORE` to feel like clear PTO interface entrypoints, not like thin aliases over one giant opaque lowering engine.
- Phase 2 should be organized so a reader can still recognize the shape of PTO template dispatch in the lowering code.
- `TLOAD` should not erase PTO operand-level features such as padding/init controls just because the first sample does not exercise them.
- `TSTORE` should visibly separate `ACC`, `VEC`, and `MAT` logic from the beginning, even if only the `VEC` path is fully implemented now.
- `TABS` should be treated as the first reusable unary-op lowering template, not just as “the `Abs` sample op.”
- The design principle for this phase is: use PTO template decision structure as the backbone so later extension does not require tearing the framework apart.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project and prior phase context
- `.planning/PROJECT.md` — project goals, compatibility constraints, and v1 scope
- `.planning/REQUIREMENTS.md` — phase requirements and acceptance boundaries
- `.planning/ROADMAP.md` — fixed Phase 2 boundary and success criteria
- `.planning/STATE.md` — current milestone position
- `.planning/phases/01-a5vm-foundation/01-CONTEXT.md` — locked Phase 1 backend decisions that Phase 2 must build on
- `.planning/phases/01-a5vm-foundation/01-RESEARCH.md` — research that established `a5vm` as a first-class dialect and locked the backend integration point
- `.planning/phases/01-a5vm-foundation/01-VALIDATION.md` — validation style and Wave 0 expectations already chosen for the backend path

### PTO IR definitions
- `include/PTO/IR/PTOOps.td` — `TLoadOp`, `TStoreOp`, and `TAbsOp` definitions, arguments, assembly, and pipe behavior

### Current lowering/backend reference
- `lib/PTO/Transforms/PTOToEmitC.cpp` — current PTO backend lowering sites for `TLOAD`, `TSTORE`, and `TABS`, useful as a reference boundary that Phase 2 is replacing
- `include/PTO/Transforms/Passes.h` — pass declaration surface for new lowering passes
- `tools/ptoas/ptoas.cpp` — current pipeline structure and backend switch location from Phase 1

### PTO library behavior references
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/common/pto_instr.hpp` — public PTO instruction template entrypoints
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TLoad.hpp` — `TLOAD_IMPL` template branching, layout checks, and valid-region inputs
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TStore.hpp` — `TSTORE_IMPL` branch structure for `VEC` / `ACC` / `MAT`, layout checks, quantization-related overloads, and valid-region inputs
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TUnaryOp.hpp` — `TABS_IMPL` behavior and unary-op restriction pattern
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TAdd.hpp` — binary op template/entrypoint pattern that helps shape reusable lowering entrypoints
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TLoad.hpp` — A5-side `TLOAD_IMPL` reference for future alignment
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TStore.hpp` — A5-side `TSTORE_IMPL` reference for future alignment
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TUnaryOp.hpp` — A5-side unary-op reference for future alignment

### Intrinsic wrapper references
- `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h` — builtin wrapper families that the lowering must eventually feed through `a5vm`

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `include/PTO/IR/PTOOps.td`: already records the PTO-side operation surface and is the source of truth for which operands and attributes exist at the IR level.
- `lib/PTO/Transforms/PTOToEmitC.cpp`: contains the current lowering touchpoints for `TLoadOp`, `TStoreOp`, and `TAbsOp`, useful for identifying where the current backend starts flattening PTO semantics too early.
- `tools/ptoas/ptoas.cpp`: already has the backend-switch boundary introduced in Phase 1, which Phase 2 lowering will plug into.
- `.planning/codebase/CONVENTIONS.md` and `.planning/codebase/STRUCTURE.md`: establish that new pass/lowering code should live under `include/PTO/Transforms/` and `lib/PTO/Transforms/`, with dialect-facing declarations under `include/PTO/IR/` if needed.

### Established Patterns
- PTO op semantics are centralized in ODS plus C++ verifiers/implementations, so lowering decisions should track those sources rather than inventing a disconnected contract.
- The repo prefers clear named pass entrypoints and adjacent helper code over scattering compiler logic across unrelated directories.
- Compiler diagnostics are expected to be technical and explicit, which fits the need for strong lowering pre-checks and unsupported-branch messaging.

### Integration Points
- Phase 2 lowering should connect the PTO op layer to the `a5vm` dialect established in Phase 1.
- The main implementation surface is likely a new lowering pass and helpers under `lib/PTO/Transforms/`, with declarations in `include/PTO/Transforms/Passes.h`.
- Future unary-op and load/store extension should reuse the Phase 2 entrypoint/helper structure instead of bypassing it with ad hoc lowering code.

</code_context>

<deferred>
## Deferred Ideas

- Broader PTO op lowering beyond the `Abs` path remains outside this phase, even though the framework should be designed to support it later.
- Full implementation of `ACC` and `MAT` `TSTORE` branches is deferred; Phase 2 should make the branches explicit and preserve their inputs, but not necessarily complete them for execution.

</deferred>

---
*Phase: 02-pto-lowering*
*Context gathered: 2026-03-18*
