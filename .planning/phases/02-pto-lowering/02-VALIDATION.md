---
phase: 02
slug: pto-lowering
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-18
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | other — MLIR `RUN:` + `FileCheck` tests plus `ctest` smoke checks |
| **Config file** | none committed in source tree; current lit config appears external or build-generated |
| **Quick run command** | `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase2/<case>.mlir -o - | FileCheck test/phase2/<case>.mlir` |
| **Full suite command** | `ctest --test-dir build --output-on-failure` |
| **Estimated runtime** | ~45 seconds |

---

## Sampling Rate

- **After every task commit:** Run `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase2/<case>.mlir -o - | FileCheck test/phase2/<case>.mlir`
- **After every plan wave:** Run `ctest --test-dir build --output-on-failure`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 45 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 0 | PTO-01 | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase2/tload_contract_trace.mlir -o - | FileCheck test/phase2/tload_contract_trace.mlir` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 0 | PTO-03 | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase2/tstore_branch_shape.mlir -o - | FileCheck test/phase2/tstore_branch_shape.mlir` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 0 | PTO-02 | unit | `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase2/tabs_precheck.mlir 2>&1 | FileCheck test/phase2/tabs_precheck.mlir` | ❌ W0 | ⬜ pending |
| 02-01-04 | 01 | 0 | PTO-04 | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase2/unary_template_shape.mlir -o - | FileCheck test/phase2/unary_template_shape.mlir` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `test/phase2/tload_contract_trace.mlir` — verify that `TLOAD` lowering preserves layout, valid region, padding/init inputs, and mapping-trace metadata needed for the contract
- [ ] `test/phase2/tstore_branch_shape.mlir` — verify that `TSTORE` lowering visibly separates `VEC`, `ACC`, and `MAT` branches and preserves branch-selection inputs
- [ ] `test/phase2/tabs_precheck.mlir` — verify `TABS` lowering-time checks for vec domain, row-major restriction, supported dtype, and matching valid shapes
- [ ] `test/phase2/unary_template_shape.mlir` — verify reusable unary lowering entrypoint shape for `TABS` and future unary ops
- [ ] A documented lit/FileCheck invocation path for `test/phase2/*` if no committed lit config is added during this phase

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| PTO template-to-lowering correspondence remains readable in code | PTO-04 | code readability of the template correspondence is partly structural, not fully machine-checkable | Inspect the lowering entrypoints and shared helpers, confirm `lowerTLOAD`, `lowerTABS`, and `lowerTSTORE` remain visible and their main PTO decision points are recognizable |
| Placeholder behavior for `ACC` / `MAT` `TSTORE` branches is explicit rather than hidden | PTO-03 | the exact quality of explicit TODO / placeholder signaling is partially qualitative | Build or inspect a case that reaches unsupported `ACC` or `MAT` lowering, confirm the branch is visible and the unsupported state is explicit |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 45s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
