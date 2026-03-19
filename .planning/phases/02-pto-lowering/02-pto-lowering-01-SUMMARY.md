---
phase: 02-pto-lowering
plan: 01
subsystem: testing
tags: [mlir, pto, a5vm, lowering, filecheck]
requires:
  - phase: 01-a5vm-foundation
    provides: "Corrected mlir::a5vm primitive surface with copy-family ops and vlds/vabs/vsts"
provides:
  - "Revised Phase 2 Wave 0 fixtures aligned to copy-family TLOAD/TSTORE and structural vec-scope TABS"
  - "Dedicated TABS precheck and backend wiring contracts for the a5vm backend path"
  - "Phase 2 runner that rejects legacy pseudo-op names and obsolete fixture references"
affects: [02-02, 02-03, phase2-verification, a5vm-backend]
tech-stack:
  added: []
  patterns: [wave-0 fixture gating, shell-level regression guards, structural a5vm lowering contracts]
key-files:
  created:
    - test/phase2/tload_copy_family_shape.mlir
    - test/phase2/tabs_abs_loop_shape.mlir
    - test/phase2/tabs_precheck_a5.mlir
    - test/phase2/tstore_copy_family_shape.mlir
    - test/phase2/tstore_domain_todos.mlir
    - test/phase2/pto_backend_a5vm_wiring.mlir
  modified:
    - test/phase2/run_phase2_checks.sh
    - test/phase2/tload_contract_trace.mlir
    - test/phase2/tabs_precheck.mlir
    - test/phase2/tstore_branch_shape.mlir
    - test/phase2/unary_template_shape.mlir
key-decisions:
  - "Gate the fixture replay on the landed mlir::a5vm primitive inventory instead of silently accepting stale Phase 1 assumptions."
  - "Model __VEC_SCOPE__ structurally in the TABS contracts by checking loop nesting and vlds/vabs/vsts ordering rather than inventing a dedicated A5VM op."
  - "Keep legacy pseudo-op and old fixture-name regressions in the shell runner so the whole Phase 2 suite can reject them before compiler execution."
patterns-established:
  - "Phase 2 contracts describe copy-family loop programming with explicit set_loop*_outtoub and set_loop*_ubtoout checkpoints."
  - "Runner-level rg guards enforce naming invariants that individual fixtures should not repeat verbatim."
requirements-completed: [PTO-01, PTO-02, PTO-03, PTO-04]
duration: 20min
completed: 2026-03-19
---

# Phase 2 Plan 01: PTO Fixture Replay Summary

**Corrected Phase 2 Wave 0 contracts for copy-family TLOAD/TSTORE, structural vec-scope TABS, and A5VM backend fixture gating**

## Performance

- **Duration:** 20 min
- **Started:** 2026-03-19T00:15:16Z
- **Completed:** 2026-03-19T00:35:16Z
- **Tasks:** 2
- **Files modified:** 11

## Accomplishments

- Replaced the obsolete Phase 2 pseudo-op fixture suite with contracts for `tload_copy_family_shape`, `tabs_abs_loop_shape`, `tabs_precheck_a5`, `tstore_copy_family_shape`, `tstore_domain_todos`, and `pto_backend_a5vm_wiring`.
- Locked the revised TABS contract to structural vec-scope lowering through loop nesting plus ordered `a5vm.vlds`, `a5vm.vabs`, and `a5vm.vsts`.
- Rewrote `test/phase2/run_phase2_checks.sh` to validate the landed A5VM primitive names, reject legacy pseudo-op regressions, and run only the revised Phase 2 fixture order before `ctest`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Gate Phase 2 on corrected Phase 1 primitives and replace the Wave 0 fixture suite** - `657d23c` (test)
2. **Task 2: Rewrite the Phase 2 runner around the corrected fixture suite** - `2c625c4` (test)

## Files Created/Modified

- `test/phase2/tload_copy_family_shape.mlir` - Contracts the corrected GM-to-UB copy family, loop programming, and preserved trace/padding metadata.
- `test/phase2/tabs_abs_loop_shape.mlir` - Locks structural vec-scope lowering to nested loops and ordered `vlds -> vabs -> vsts`.
- `test/phase2/tabs_precheck_a5.mlir` - Captures pre-A5VM diagnostics for unsupported domain, layout, valid-region, and dtype cases.
- `test/phase2/tstore_copy_family_shape.mlir` - Contracts UB-to-GM copy-family lowering and visible `set_loop*_ubtoout` programming.
- `test/phase2/tstore_domain_todos.mlir` - Preserves visible `VEC`, `ACC`, and `MAT` source-domain behavior plus TODO diagnostics.
- `test/phase2/pto_backend_a5vm_wiring.mlir` - Proves the `--pto-backend=a5vm` path lowers PTO ops before final backend emission.
- `test/phase2/run_phase2_checks.sh` - Runs only the revised Phase 2 fixture suite and rejects obsolete pseudo-op regressions.

## Decisions Made

- Gated the replay on `mlir::a5vm` plus `CopyGmToUbuf`, `CopyUbufToGm`, `Vlds`, `Vabs`, and `Vsts` so Phase 2 fixtures now fail fast if Phase 1 drifts.
- Represented `__VEC_SCOPE__` structurally in the fixture contract instead of encoding a fake A5VM op, matching the user-selected revised planning.
- Centralized the legacy-name regression check in the runner so fixture files themselves can stay focused on the corrected target shape.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed legacy pseudo-op spellings from backend wiring fixture comments**
- **Found during:** Task 1 (Gate Phase 2 on corrected Phase 1 primitives and replace the Wave 0 fixture suite)
- **Issue:** The initial backend wiring fixture still contained `CHECK-NOT` comments with obsolete pseudo-op names, which violated the revised fixture ban on those spellings anywhere under `test/phase2`.
- **Fix:** Removed the stale comment checks and left the obsolete-name guard in the runner instead.
- **Files modified:** `test/phase2/pto_backend_a5vm_wiring.mlir`
- **Verification:** Re-ran the Task 1 primitive-and-fixture gate; the suite no longer matches the forbidden-name grep.
- **Committed in:** `657d23c` (part of task commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** The fix kept the replay aligned with the revised no-legacy-name contract without changing scope.

## Issues Encountered

- The plan’s automated checks for this replay are file- and grep-based only. I did not run the full `ptoas | FileCheck` suite because the later implementation plans are the ones that make those new contracts executable.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 plan `02-02` can now build the helper/lowering layer against the corrected Wave 0 contract instead of the superseded pseudo-op suite.
- The runner and fixture names now match the revised planning docs, so future validation can target the correct files directly.

## Self-Check: PASSED

- Found summary file `.planning/phases/02-pto-lowering/02-pto-lowering-01-SUMMARY.md`
- Found task commit `657d23c`
- Found task commit `2c625c4`

---
*Phase: 02-pto-lowering*
*Completed: 2026-03-19*
