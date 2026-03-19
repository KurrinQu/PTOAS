---
phase: 02-pto-lowering
plan: 04
subsystem: compiler
tags: [a5vm, lowering, memref, filecheck, pto]
requires:
  - phase: 02-pto-lowering
    provides: explicit PTO-to-A5VM helper lowering and backend wiring for TLOAD/TSTORE
provides:
  - recovers copy-family stride metadata through memref adapter chains
  - recovers partition trace offsets and sizes from lowered subview form
  - locks exact recovered stride and trace attrs in Phase 2 copy-family fixtures
affects: [phase-2-verification, tload, tstore, a5vm]
tech-stack:
  added: []
  patterns:
    - recursive metadata recovery through memref adapter chains
    - fixture-locked copy-family lowering attrs validated from emitted IR
key-files:
  created: []
  modified:
    - include/PTO/Transforms/A5VMLowering.h
    - lib/PTO/Transforms/PTOToA5VMLowering.cpp
    - test/phase2/tload_copy_family_shape.mlir
    - test/phase2/tstore_copy_family_shape.mlir
key-decisions:
  - "Recover copy-family metadata by walking memref.subview, memref.reinterpret_cast, and memref.cast instead of relying on direct PartitionViewOp producers."
  - "Keep Task 2 focused on fixture contracts by locking only the exact stride and trace attrs the emitted IR now proves."
patterns-established:
  - "Copy-family metadata recovery: compose subview offsets and sizes onto existing partition trace while preserving tensor-view stride facts."
  - "Fresh-build verification fallback: use a clean throwaway build when the checked-in build directory is inconsistent."
requirements-completed: [PTO-01, PTO-03, PTO-04]
duration: 75min
completed: 2026-03-19
---

# Phase 2 Plan 4: Copy-Family Metadata Summary

**Recovered exact copy-family stride and partition-trace attrs for TLOAD/TSTORE through memref adapter chains and locked them in Phase 2 fixtures**

## Performance

- **Duration:** 75 min
- **Started:** 2026-03-19T02:37:08Z
- **Completed:** 2026-03-19T03:52:08Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Recovered `src_strides` / `dst_strides` from `memref.subview`, `memref.reinterpret_cast`, and `memref.cast` chains instead of degrading to dynamic sentinels.
- Recovered `trace_offsets` and `trace_sizes` from the lowered subview form that reaches Phase 2 A5VM lowering.
- Tightened both copy-family fixtures to require the exact recovered stride metadata alongside the existing loop-programming and trace checks.

## Task Commits

Each task was committed atomically:

1. **Task 1: Recover exact partition trace and tensor-view strides through the lowering-time value chain** - `ba61ea6` (fix)
2. **Task 2: Lock the exact recovered metadata in the copy-family fixtures** - `0683a9e` (test)

## Files Created/Modified
- `include/PTO/Transforms/A5VMLowering.h` - Adds the missing loop interface include exposed by the fresh-build verification path.
- `lib/PTO/Transforms/PTOToA5VMLowering.cpp` - Walks memref adapter chains to recover tensor-view strides and partition trace metadata at lowering time.
- `test/phase2/tload_copy_family_shape.mlir` - Requires `src_strides = [32, 1]` on lowered `a5vm.copy_gm_to_ubuf`.
- `test/phase2/tstore_copy_family_shape.mlir` - Requires `dst_strides = [32, 1]` on lowered `a5vm.copy_ubuf_to_gm`.

## Decisions Made
- Recover metadata from the actual lowered SSA graph instead of special-casing the direct `pto.partition_view` form.
- Preserve the fixture structure and add only the missing stride assertions needed by the verification gap.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added missing loop interface include for clean A5VM helper builds**
- **Found during:** Task 1 (Recover exact partition trace and tensor-view strides through the lowering-time value chain)
- **Issue:** A fresh build failed because `A5VMLowering.h` declared `LoopLikeOpInterface` without including `mlir/Interfaces/LoopLikeInterface.h`.
- **Fix:** Added the missing interface include in the public lowering header.
- **Files modified:** `include/PTO/Transforms/A5VMLowering.h`
- **Verification:** Fresh `/tmp/ptoas-02-04-build` compilation proceeded past `PTOToA5VM.cpp` and `PTOToA5VMLowering.cpp`.
- **Committed in:** `ba61ea6` (part of task commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** The deviation was required to compile the planned lowering changes in a clean environment. No scope creep.

## Issues Encountered
- The checked-in `build/` directory was internally inconsistent for this task: `ninja` looped on CMake regeneration, omitted A5VM-generated headers from the graph, and lacked a workspace `FileCheck` binary. Verification was completed with a clean throwaway build at `/tmp/ptoas-02-04-build` and direct emitted-IR assertions against the plan-required attrs.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 2 copy-family lowering now preserves the exact stride and trace metadata required by backend selection in observable emitted IR.
- The remaining Phase 2 work is still centered on vec-scope carrier visibility, locked diagnostics, and a workspace-safe verification runner.

## Self-Check: PASSED

---
*Phase: 02-pto-lowering*
*Completed: 2026-03-19*
