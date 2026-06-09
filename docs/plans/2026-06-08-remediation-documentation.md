# Documentation & Metrics Unification — Implementation Plan

> **For Agent:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** Unify metrics, thresholds, and performance narratives across the entire repository to resolve the three incompatible narratives that pose a direct credibility risk.

**Architecture:**
1. Declare `reports/end_to_end_optimized_results.json` as the single canonical source of truth for all post-optimization metrics (F1, Precision, Recall, Latency).
2. Reconcile the decision threshold records between the actual code artifact (`models/optimal_threshold_v2.json` -> 0.76) and markdown documents.
3. Correct all stale figures in `README.md` and `docs/business_impact.md` to match the canonical JSON.
4. Correct CI/CD references to reflect the actual pipeline config, resolving the external reviewer's incorrect claim.

**Tech Stack:** Markdown · JSON · Python

**Design Decisions / Audit Reference:**
- Production Audit Report: P0-2, P0-3, P1-5, P1-6, P2-3
- Claims Validity Matrix: Item 3, Item 7, Item 9, Item 10

---

## Mandatory Gate

> ⚠️ **Do not begin this remediation plan until the implementation plan itself is approved by the user.**
> ⚠️ **All tasks must end with a single semantic git commit. Do not squash multiple tasks into one commit.**

---

### Task 1: Reconcile Decision Threshold Records (0.76 vs 0.42)

**Files:**
- Modify: [`docs/model_architecture.md`](file:///d:/Projects/ai-ml/creditcard-fraud/docs/model_architecture.md)

**Step 1: Check Current Model Artifacts**
Confirm that the active optimized threshold in `models/optimal_threshold_v2.json` is `0.76`.

**Step 2: Update model_architecture.md**
Search for references to `0.42` threshold in `docs/model_architecture.md` and update them to `0.76` to match the actual deployed configuration and the optimized output logs. If necessary, clarify that `0.42` was a baseline threshold and `0.76` is the finalized, optimized decision boundary.

**Step 3: Commit**
```bash
git add docs/model_architecture.md
git commit -m "docs: reconcile decision threshold records to match canonical 0.76"
```

---

### Task 2: Unify Metrics in `README.md` and `docs/business_impact.md`

**Files:**
- Modify: `README.md`
- Modify: [`docs/business_impact.md`](file:///d:/Projects/ai-ml/creditcard-fraud/docs/business_impact.md)

**Step 1: Reference Canonical Source**
Extract the exact metrics (F1, Precision, Recall, Latency p95, Average Latency) from [`reports/end_to_end_optimized_results.json`](file:///d:/Projects/ai-ml/creditcard-fraud/reports/end_to_end_optimized_results.json).

**Step 2: Edit README.md**
Update the evaluation tables and text in `README.md` to use the canonical numbers. Ensure no conflicting baseline metrics are presented as the "optimized" performance without clear categorization.

**Step 3: Edit docs/business_impact.md**
Update all business impact calculations, financial metrics, and performance narratives to use the exact F1, Precision, and Recall values from the canonical JSON report. Ensure consistency in the simulated savings or cost-benefit analysis tables.

**Step 4: Commit**
```bash
git add README.md docs/business_impact.md
git commit -m "docs: unify performance metrics across README and business impact reports"
```

---

### Task 3: Add Metadata Fields to All JSON Reports

**Files:**
- Modify: Any JSON files in `reports/*.json`

**Step 1: Inject Metadata Keys**
Inject standard metadata fields to all canonical metric reports to trace execution origin:
- `"generated_at"`: UTC timestamp of execution.
- `"source_script"`: Path of the script that produced the JSON.
- `"schema_version"`: Set to `1`.

Example for `reports/end_to_end_optimized_results.json`:
```json
{
  "schema_version": 1,
  "generated_at": "2026-06-08T15:00:00Z",
  "source_script": "debug_scripts/end_to_end_test_optimized.py",
  "metrics": { ... }
}
```

**Step 2: Commit**
```bash
git add reports/*.json
git commit -m "chore: add schema metadata to all JSON report files"
```

---

### Task 4: Explicitly Correct CI/CD Documentation Claims

**Files:**
- Modify: `README.md` or a central docs page

**Step 1: Document CI/CD Setup**
Verify that the `README.md` clearly describes the CI/CD pipeline setup (`.github/workflows/ci.yml`). Add a section highlighting the 4-job automated suite:
- Linting (flake8/black)
- Unit tests (pytest)
- Pipeline configuration validations
- Docker image building

This explicitly corrects the external reviewer's false assertion that CI/CD is missing from the repository.

**Step 2: Commit**
```bash
git add README.md
git commit -m "docs: add CI/CD pipeline description to README"
```
