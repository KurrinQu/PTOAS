---
phase: 02-pto-lowering
plan: 02
subsystem: api
tags: [mlir, a5vm, pto, lowering, scf, cmake]
requires:
  - phase: 02-pto-lowering
    provides: Revised 02-01 fixture contracts and phase runner for the corrected PTO-to-A5VM shape
provides:
  - Public A5-only lowering contracts for TLOAD, TABS, and TSTORE
  - Shared PTOToA5VMLowering helper layer for contract extraction, copy-loop programming, and unary vec-scope construction
  - Build-wired SCF-enabled Phase 2 lowering path with explicit TSTORE ACC and MAT TODO diagnostics
affects: [02-03-PLAN.md, ptoas, a5vm]
tech-stack:
  added: []
  patterns: [Explicit lowering contracts, helper-based PTO semantic extraction, SCF vec-scope lowering, attribute-backed copy loop programming]
key-files:
  created:
    - lib/PTO/Transforms/PTOToA5VMLowering.cpp
  modified:
    - include/PTO/Transforms/A5VMLowering.h
    - lib/PTO/Transforms/PTOToA5VM.cpp
    - lib/PTO/Transforms/CMakeLists.txt
    - include/PTO/Transforms/Passes.td
key-decisions:
  - "Keep the public Phase 2 surface limited to lowerTLOAD, lowerTABS, and lowerTSTORE plus truthful A5-only contracts."
  - "Represent copy-family set_loop programming as explicit attached metadata so the PTO branch structure stays visible before dedicated loop ops exist."
  - "Build unary Abs lowering as structural SCF vec-scope loops around vlds, vabs, and vsts, and register SCF in the pass dependency list."
patterns-established:
  - "Extract PTO lowering contracts first, then materialize A5VM operations from those contracts."
  - "Materialize memref views for pointer-backed or tile-backed values before building A5VM copy/vector ops."
requirements-completed: [PTO-01, PTO-02, PTO-03, PTO-04]
duration: 24min
completed: 2026-03-19
---

# Phase 2 Plan 2: PTO Lowering Summary

**A5-only PTO lowering contracts with a build-wired helper layer for copy-family loop metadata and structural unary vec-scope lowering**

## Performance

- **Duration:** 24 min
- **Started:** 2026-03-19T00:34:00Z
- **Completed:** 2026-03-19T00:58:49Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Rewrote the public lowering header to expose only the corrected A5-only contracts and explicit `lowerTLOAD`, `lowerTABS`, and `lowerTSTORE` entrypoints.
- Split the PTO-to-A5VM implementation into a dedicated helper translation unit that extracts truthful PTO contracts, preserves partition/valid-shape metadata, and programs copy-family loop attributes explicitly.
- Wired the helper layer into `PTOTransforms`, registered `scf` for unary vec-scope lowering, rebuilt `ptoas`, and validated the `tabs_abs_loop_shape` fixture against the rebuilt CLI.

## Task Commits

Each task was committed atomically:

1. **Task 1: Redefine the public Phase 2 lowering contracts around A5-only semantics** - `ded4b7d` (feat)
2. **Task 2: Implement the shared helper layer that mirrors PTO-library control structure** - `3c5969e` (feat)

## Files Created/Modified

- `include/PTO/Transforms/A5VMLowering.h` - Public A5-only contracts, entrypoints, and shared helper declarations for copy-loop programming and unary vec-scope construction.
- `lib/PTO/Transforms/PTOToA5VMLowering.cpp` - Shared contract extraction, memref materialization, copy-loop metadata programming, and structural `vlds -> vabs -> vsts` lowering.
- `lib/PTO/Transforms/PTOToA5VM.cpp` - Pass wiring reduced to dispatch and erase semantics around the extracted helper entrypoints.
- `lib/PTO/Transforms/CMakeLists.txt` - Adds the helper translation unit and SCF dialect linkage to `PTOTransforms`.
- `include/PTO/Transforms/Passes.td` - Registers `scf::SCFDialect` so structural vec-scope loops can be built at runtime.

## Decisions Made

- Kept the public boundary narrow: contracts and explicit PTO entrypoints are public, while extraction and lowering mechanics live in `PTOToA5VMLowering.cpp`.
- Stored `set_loop*_outtoub` and `set_loop*_ubtoout` semantics as readable attached attributes on copy ops, which preserves the PTO decision structure without inventing new dialect ops in this plan.
- Lowered `TABS` through structural `scf.for` nesting plus ordered `a5vm.vlds`, `a5vm.vabs`, and `a5vm.vsts`, which matches the corrected Phase 2 fixture contract and leaves room for future unary reuse.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Materialized memref views for pointer-backed and tile-backed operands**
- **Found during:** Task 2
- **Issue:** The A5VM copy/vector ops require memref operands, while Phase 2 sources and destinations can arrive as PTO pointers or tile-annotated values.
- **Fix:** Added helper-level memref materialization before creating A5VM copy or vector ops.
- **Files modified:** `lib/PTO/Transforms/PTOToA5VMLowering.cpp`
- **Verification:** `env CCACHE_DISABLE=1 cmake --build build --target pto-opt -j2`
- **Committed in:** `3c5969e`

**2. [Rule 3 - Blocking] Registered SCF as a pass dependency for unary vec-scope lowering**
- **Found during:** Task 2
- **Issue:** The rebuilt CLI failed at runtime because `scf.for` was constructed without `SCF` being registered in the pass dependency list.
- **Fix:** Added `scf::SCFDialect` to the pass definition and included the SCF dialect header in the pass wiring TU.
- **Files modified:** `include/PTO/Transforms/Passes.td`, `lib/PTO/Transforms/PTOToA5VM.cpp`
- **Verification:** `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase2/tabs_abs_loop_shape.mlir -o /dev/null 2>&1 | /data/mouliangyu/projects/github.com/llvm/llvm-project/build/bin/FileCheck test/phase2/tabs_abs_loop_shape.mlir`
- **Committed in:** `3c5969e`

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both fixes were required to make the planned helper layer compile and execute in the real CLI. No architectural scope change.

## Issues Encountered

- `CLAUDE.md` was not present at the repository root, so execution proceeded from the phase planning artifacts and workspace skill index.
- Sandbox builds initially failed because `ccache` attempted to write to a read-only home directory path. Verification succeeded with `CCACHE_DISABLE=1`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan `02-03` can now wire these helper entrypoints into the remaining backend path knowing the contracts, helper layer, and SCF vec-scope structure already exist.
- The current helper layer preserves the semantic inputs and branch visibility needed for future completion of `TSTORE` ACC/MAT branches and broader unary reuse.

## Self-Check: PASSED

---
*Phase: 02-pto-lowering*
*Completed: 2026-03-19*
