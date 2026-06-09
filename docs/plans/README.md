# Implementation Plans Index

> Source design: [`docs/design_decisions.md`](../design_decisions.md)  
> Audit disposition: **APPROVED WITH MANDATORY CONDITIONS** (2026-06-08)

These plans are the executable breakdown of the 4-phase production backend defined in the
Multi-Agent Brainstorming audit. Execute phases **in order** — each phase is a hard prerequisite
for the next.

---

## Execution Order

| # | Plan File | Phase | Mandatory Pre-condition Gate |
|---|---|---|---|
| 1 | [`2026-06-08-phase1-foundation.md`](2026-06-08-phase1-foundation.md) | Foundation & Infrastructure | Must pass before any code is written |
| 2 | [`2026-06-08-phase2-inference-path.md`](2026-06-08-phase2-inference-path.md) | Inference API | All Phase 1 tasks marked `[x]` |
| 3 | [`2026-06-08-phase3-observability.md`](2026-06-08-phase3-observability.md) | Observability & Streaming | `/predict` endpoint live and passing tests |
| 4 | [`2026-06-08-phase4-business-intelligence.md`](2026-06-08-phase4-business-intelligence.md) | Business Intelligence | Phase 3 WebSocket endpoint live |

---

## Remediation Plans

These plans detail the steps to fix the issues identified in the Production Audit Report (dated 2026-06-08).

| # | Plan File | Focus Area | Goal |
|---|---|---|---|
| 1 | [`2026-06-08-remediation-ml-science.md`](2026-06-08-remediation-ml-science.md) | ML Science & Optimization | Fix threshold leakage, temporal CV split, Optuna scale. |
| 2 | [`2026-06-08-remediation-documentation.md`](2026-06-08-remediation-documentation.md) | Documentation & Metrics | Unify performance narratives and decision thresholds. |
| 3 | [`2026-06-08-remediation-pipeline-infrastructure.md`](2026-06-08-remediation-pipeline-infrastructure.md) | Pipeline Integrity & Tech Debt | Fix overwrite race conditions, latency metrics, add tests. |

---

## How to Execute

**Option A — Subagent-Driven (this session):**
Open each plan file and run: use `/executing-plans` skill to implement task-by-task with review
between each task.

**Option B — Parallel Session:**
Open a new session, reference a plan file, and use `/executing-plans` to batch through it
with checkpoint commits.

> **Rule:** Each task ends with a `git commit`. Never batch more than one task into a single commit.
