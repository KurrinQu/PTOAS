---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 2
current_phase_name: PTO Lowering
current_plan: 2
status: in_progress
stopped_at: Completed 02-01-PLAN.md
last_updated: "2026-03-19T00:35:16Z"
last_activity: 2026-03-19
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 10
  completed_plans: 4
  percent: 40
---

# Project State

**Updated:** 2026-03-19
**Status:** In progress — revised Phase 2 replay underway
**Current Phase:** 2
**Current Phase Name:** PTO Lowering
**Total Phases:** 4
**Current Plan:** 2
**Total Plans in Phase:** 3
**Progress:** [████░░░░░░] 40%
**Last Activity:** 2026-03-19
**Last Activity Description:** Replayed revised 02-01 Phase 2 fixture and runner contracts

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-03-18)

**Core value:** Preserve PTO library semantics and template-driven behavior inside PTOAS so backend lowering retains enough information to enable optimization instead of losing it during library instantiation.
**Current focus:** Phase 2 - PTO Lowering

## Current Position

- Project initialized
- Workflow preferences captured
- Research completed
- Requirements defined
- Roadmap created
- Phase 1 plans created
- Plan `01-01` executed and summarized
- Plan `01-02` executed and summarized
- Plan `01-03` executed and summarized
- Revised plan `02-01` executed and summarized
- Next execution target: `02-02-PLAN.md`

## Active Milestone

**Name:** Initial A5VM backend bring-up
**Goal:** Replace the `emitc` backend slot with a PTOAS-native `a5vm` path that can compile the `Abs` sample and produce textual LLVM HIVM intrinsic IR.

## Phase Status

| Phase | Name | Status |
|-------|------|--------|
| 1 | A5VM Foundation | Complete |
| 2 | PTO Lowering | In Progress |
| 3 | HIVM Emission | Pending |
| 4 | Abs Validation | Pending |

## Requirements Snapshot

- Total v1 requirements: 16
- Complete: 10
- In Progress: 0
- Pending: 6
- Blocked: 0

## Key Decisions Snapshot

- Introduce `a5vm` as the hardware-facing backend dialect.
- Replace the current `emitc` slot rather than redesigning the pass pipeline.
- Keep v1 limited to the `Abs` sample and the minimum PTO interface set it requires.
- Emit textual LLVM HIVM intrinsic IR first, then confirm final intrinsic spellings externally.
- Use committed MLIR `RUN:`/`FileCheck` fixtures as the Phase 1 backend contract before implementation starts.
- Use a standalone Bash runner for Phase 1 verification instead of relying on external lit configuration.
- Use a handwritten A5VM vector type parser/printer to preserve the exact `!a5vm.vec<...>` syntax under the local MLIR toolchain.
- Keep `emitc` as the default backend while exposing `a5vm` through an explicit `--pto-backend` selector.
- Treat raw A5VM textual fixtures as already-lowered backend IR on the A5VM path so debug IR preserves shared dialects and A5VM ops.
- Report unsupported A5VM seam cases through explicit comments, diagnostics, and optional sidecar files instead of guessing later-stage emission behavior.
- Use committed Phase 2 MLIR/FileCheck fixtures as the PTO semantic-lowering contract before implementing the lowering pass.
- Use a standalone Bash runner for Phase 2 verification instead of relying on external lit configuration.

## Recent Progress

- Refreshed `.planning/phases/02-pto-lowering/02-pto-lowering-01-SUMMARY.md` against the revised planning docs after the decision checkpoint
- Replaced the obsolete Phase 2 pseudo-op fixture suite with corrected `tload_copy_family_shape`, `tabs_abs_loop_shape`, `tabs_precheck_a5`, `tstore_copy_family_shape`, `tstore_domain_todos`, and `pto_backend_a5vm_wiring` contracts
- Rewrote `test/phase2/run_phase2_checks.sh` to gate on landed A5VM primitive names and reject legacy pseudo-op regressions before compiler execution
- Marked PTO-01 through PTO-04 complete in `.planning/REQUIREMENTS.md`

## Open Questions

- Which exact LLVM HIVM intrinsic spellings correspond to each builtin variant exercised by the final `Abs` path
- Whether the implemented `Abs` path needs only the currently expected load/abs/store intrinsic families or additional helper intrinsics

## Session Continuity

- Next recommended command: `/gsd:execute-phase 02-pto-lowering`
- Next plan to execute: `02-02-PLAN.md`
- Current blocker status: none

## Performance Metrics

| Phase | Duration | Tasks | Files |
|-------|----------|-------|-------|
| Phase 02-pto-lowering P01 (replay) | 20min | 2 tasks | 11 files |
| Phase 01-a5vm-foundation P03 (refresh) | 22min | 2 tasks | 4 files |
| Phase 02 P02 | 7min | 2 tasks | 3 files |
| Phase 02-pto-lowering P03 | 24min | 2 tasks | 6 files |
| Phase 01 P01 | 21min | 2 tasks | 10 files |
| Phase 01-a5vm-foundation P02 | 25min | 2 tasks | 8 files |

## Decisions Made


- [Phase 02]: Keep the lowering surface split into public contracts plus a helper implementation file before pass wiring.
- [Phase 02]: Use explicit metadata attachment helpers so fixture-locked attribute names stay readable and reusable.
- [Phase 02]: Preserve unsupported TSTORE ACC and MAT paths as dedicated TODO diagnostics instead of collapsing them into a generic failure.
- [Phase 02]: Run PTO-to-A5VM only on the --pto-backend=a5vm branch after the shared pre-backend passes.
- [Phase 02]: Extract tile layout, valid dims, and address-space metadata from bind_tile and pointer_cast SSA chains because the A5VM boundary sees memref-backed tile values.
- [Phase 02]: Use an explicit rewrite walk instead of greedy pattern application so single-op Phase 2 fixtures retain visible a5vm.load and a5vm.abs ops in debug IR.
- [Phase 02]: Gate revised Wave 0 fixture replay on the landed A5VM primitive names instead of silently tolerating stale Phase 1 assumptions.
- [Phase 02]: Represent `__VEC_SCOPE__` structurally in Phase 2 fixtures by checking loop nesting and ordered `vlds` / `vabs` / `vsts`.
- [Phase 02]: Keep obsolete pseudo-op name rejection in the Phase 2 runner so fixture files stay focused on the corrected lowering shape.
- [Phase 01-a5vm-foundation]: Keep the Phase 1 A5VM seam at raw corrected backend text and defer llvm.hivm emission to the later HIVM phase.
- [Phase 01]: Keep the no-legacy-name regression check in the standalone runner rather than in the MLIR fixtures so file-level validation can forbid obsolete spellings entirely.
- [Phase 01-a5vm-foundation]: Keep copy-op transfer attrs parser-optional and verifier-required so invalid fixtures fail with the planned diagnostic instead of a parser error.
- [Phase 01-a5vm-foundation]: Derive copy transfer metadata from existing lowering contract fields instead of widening the public Phase 2 lowering structs in this plan.
- [Phase 01-a5vm-foundation]: Add A5VMOpsIncGen as a direct ptoas build dependency because the CLI includes generated A5VM headers before linking against PTOIR.

## Blockers

None.

## Session

**Last Date:** 2026-03-18T20:35:30.003Z
**Stopped At:** Completed 02-01-PLAN.md
**Resume File:** None

---
*Last updated: 2026-03-19 after completing revised plan 02-01*
