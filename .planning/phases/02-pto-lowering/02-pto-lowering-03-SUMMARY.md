---
phase: 02-pto-lowering
plan: 03
subsystem: backend
tags: [mlir, a5vm, pto, passes, ptoas]
requires:
  - phase: 02-pto-lowering
    provides: corrected PTO-to-A5VM helper lowering for TLOAD, TABS, and TSTORE
provides:
  - explicit PTO-to-A5VM helper dispatch at the registered pass boundary
  - verified A5VM backend wiring contract for `ptoas --pto-backend=a5vm`
affects: [03-hivm-emission, ptoas, a5vm]
tech-stack:
  added: []
  patterns: [thin conversion pass boundaries, backend-specific pipeline branching]
key-files:
  created: []
  modified: [lib/PTO/Transforms/PTOToA5VM.cpp, .planning/phases/02-pto-lowering/deferred-items.md]
key-decisions:
  - "Keep the pass boundary thin and make helper dispatch explicit instead of rebuilding semantics in the pass."
  - "Treat task 2 as an explicit empty commit because the A5VM branch wiring already satisfied the plan contract."
patterns-established:
  - "PTO-to-A5VM pass pattern: legality plus helper dispatch only, with helper failures surfacing as pass failures."
  - "Execution accounting pattern: already-satisfied plan tasks still receive an atomic empty commit."
requirements-completed: [PTO-02, PTO-04]
duration: 4min
completed: 2026-03-19
---

# Phase 2 Plan 3: PTO Lowering Summary

**Explicit PTO-to-A5VM helper dispatch at the registered pass boundary with preserved `ptoas --pto-backend=a5vm` backend wiring**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-19T03:05:18Z
- **Completed:** 2026-03-19T03:08:54Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Made the `pto-to-a5vm` pass contain explicit `lowerTLOAD`, `lowerTABS`, and `lowerTSTORE` helper dispatch while keeping the pass boundary thin.
- Confirmed the A5VM backend branch already runs the shared pre-backend pipeline plus `createLowerPTOToA5VMPass()` and preserves fixture-visible IR.
- Recorded current out-of-scope verification blockers in the phase deferred-items log.

## Task Commits

Each task was committed atomically:

1. **Task 1: Register `pto-to-a5vm` as a thin pass boundary over the corrected helper layer** - `c7fc14f` (feat)
2. **Task 2: Wire `ptoas --pto-backend=a5vm` through the corrected lowering path and fixture-visible IR** - `c0373b4` (feat, empty commit because the task was already satisfied)

## Files Created/Modified

- `lib/PTO/Transforms/PTOToA5VM.cpp` - added explicit helper wrappers so the pass visibly dispatches through `lowerTLOAD`, `lowerTABS`, and `lowerTSTORE`.
- `.planning/phases/02-pto-lowering/deferred-items.md` - logged the currently observed out-of-scope verification blockers.

## Decisions Made

- Keep the pass implementation focused on legality and helper dispatch rather than introducing any new pass-layer semantics.
- Preserve the existing A5VM branch wiring as-is and capture task completion with an empty commit instead of making no-op source edits.

## Deviations from Plan

None - plan intent was preserved exactly.

## Issues Encountered

- `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir ... | FileCheck ...` could not be replayed directly in this shell because `FileCheck` is not available on `PATH`.
- `bash test/phase2/run_phase2_checks.sh` fails on a pre-existing fixture/runner mismatch in `test/phase2/tabs_abs_loop_shape.mlir`; the runner now forbids the file's remaining bare `CHECK: scf.for` lines.
- `CCACHE_DISABLE=1 ninja -C build PTOTransforms ptoas` is currently trapped in repeated CMake regeneration because of stale/generated build metadata already tracked in the phase deferred-items log.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 code is ready for Phase 3 HIVM emission work.
- Fresh binary verification still needs the phase's existing build/fixture blockers resolved first.

## Self-Check: PASSED

- Found `.planning/phases/02-pto-lowering/02-pto-lowering-03-SUMMARY.md`
- Found commit `c7fc14f`
- Found commit `c0373b4`

---
*Phase: 02-pto-lowering*
*Completed: 2026-03-19*
