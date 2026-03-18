# Stack Research

## Scope

Project-specific implementation stack for replacing the current `emitc` backend with a PTOAS-native `a5vm` backend that only needs to satisfy the `Abs` sample in v1.

## Recommended Stack

### Frontend / Source IR

- **PTO dialect**: Keep existing PTO IR as the semantic source of truth.
  Confidence: High
  Rationale: The user explicitly requires existing PTO interface semantics and template behavior to stay aligned.

- **Shared MLIR dialects (`func`, `scf`, `arith`, `cf`)**: Keep existing use for control flow and scalar logic.
  Confidence: High
  Rationale: The user explicitly wants ordinary control flow and scalar arithmetic to remain in general dialects.

### PTO-Library Semantic Layer

- **Dedicated PTO-to-A5VM lowering framework**: Add a new lowering layer that handles PTO interfaces such as `TLOAD`, `TABS`, and `TSTORE`.
  Confidence: High
  Rationale: This is the layer that preserves PTO template-driven intent before it is flattened into backend-specific operations.

- **TableGen-driven or helper-driven lowering metadata**: Reuse the same type/layout/shape decision points the existing PTO backend logic already depends on.
  Confidence: Medium
  Rationale: v1 only needs a narrow slice, but the framework should be able to encode template-derived variants instead of hardcoding a one-off `Abs` path.

### Hardware-Facing Dialect

- **New `a5vm` dialect** with:
  - legal fixed-length 256-byte vector types
  - ops for the minimum `Abs` path, starting with load, abs, and store style vector operations
  - attributes/enums for distribution/layout variants that determine intrinsic spelling
  Confidence: High
  Rationale: Separates PTO semantics from hardware/intrinsic codegen and gives the backend a stable IR boundary.

### Final Emission Layer

- **Textual LLVM HIVM intrinsic emission** instead of `emitc`
  Confidence: High
  Rationale: The target intrinsic set is not fully available in the local LLVM environment, so v1 must produce structurally valid textual IR as the backend artifact.

- **Intrinsic name synthesis helpers** derived from builtin wrapper shape
  Confidence: High
  Rationale: Local Ascend headers show the builtin wrapper families encode variants in the name itself, for example `vldsx1`, lane count, and element type suffixes.

## Key Local References

- `lib/PTO/Transforms/PTOToEmitC.cpp`
  Why it matters: Current backend insertion point and existing PTO op lowering registry.

- `test/samples/Abs/abs.py`
  Why it matters: Defines the v1 acceptance path and the exact PTO ops exercised.

- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/common/pto_instr.hpp`
  Why it matters: Defines public PTO instruction templates such as `TABS`, `TLOAD`, and `TSTORE`.

- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TUnaryOp.hpp`
  Why it matters: Shows that `TABS_IMPL` eventually dispatches to `_vabs` and then `vabs(...)`.

- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TLoad.hpp`
  Why it matters: Shows how `TLOAD_IMPL` selects the GM-to-Vec path and what layout constraints it enforces.

- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TStore.hpp`
  Why it matters: Shows how `TSTORE_IMPL` dispatches by source tile domain and validates shape/layout/dtype constraints.

- `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h`
  Why it matters: Defines wrapper families like `vld`, `vlds`, `vst`, and their underlying `__builtin_cce_*` spellings.

## What Not To Use In v1

- **The current `emitc` C++ emission path**: It is the system being replaced at this pipeline boundary.
- **A full LLVM intrinsic integration that depends on locally registered HIVM intrinsics**: Not available in the current environment.
- **A broad PTO library reimplementation**: v1 should stay constrained to the minimum `Abs` slice.

## Current Recommendation

Build the v1 backend as:

`PTO -> PTO-to-A5VM lowering -> A5VM dialect -> textual LLVM HIVM intrinsic emission`

This preserves semantic structure longer than the current library-instantiation path and directly targets the user’s stated optimization problem.
