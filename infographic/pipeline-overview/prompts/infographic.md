Create a professional infographic following these specifications:

## Image Specifications

- **Type**: Infographic
- **Layout**: comparison-matrix
- **Style**: corporate-memphis
- **Aspect Ratio**: landscape
- **Language**: en

## Core Principles

- Follow the layout structure precisely for information architecture
- Apply style aesthetics consistently throughout
- If content involves sensitive or copyrighted figures, create stylistically similar alternatives
- Keep information concise, highlight keywords and core concepts
- Use ample whitespace for visual clarity
- Maintain clear visual hierarchy

## Text Requirements

- All text must match the specified style treatment
- Main titles should be prominent and readable
- Key concepts should be visually emphasized
- Labels should be clear and appropriately sized
- Use the specified language for all text content

## Layout Guidelines

Grid-based multi-factor comparison across multiple items.

- Table/grid layout
- Rows: items being compared
- Columns: comparison criteria
- Cells: scores, checks, or values
- Header row and column clearly marked

## Style Guidelines

Flat vector people with vibrant geometric fills

- Color Palette: Bright, saturated - purple, orange, teal, yellow; Background: White or light pastels; Accents: Gradient fills, geometric patterns
- Visual Elements: Flat vector illustration, disproportionate human figures, abstract body shapes, floating geometric elements, no outlines, solid fills, plant and object accents
- Typography: Clean sans-serif, bold headings, professional but friendly, minimal decoration

---

## Infographic Content

**Title**: Real-Time Credit Card Fraud Detection Pipeline

**Subtitle**: Baseline vs Optimized LightGBM Model — Production-Grade ML with Strict Temporal Validation

---

### Grid Structure

| Criterion | Baseline Model | Optimized Model | Target | Status |
|-----------|---------------|-----------------|--------|--------|
| F1-Score | 0.8041 | 0.8478 | > 0.85 | NEAR TARGET |
| Precision | 0.8667 | 0.9750 | > 0.90 | PASS |
| Recall | 0.7500 | 0.7500 | > 0.80 | NEAR TARGET |
| ROC AUC | 0.9748 | 0.9739 | — | EXCELLENT |
| Mean Latency | 1.40 ms | 3.03 ms | — | OK |
| 95th Percentile Latency | 3.63 ms | 8.89 ms | < 10.00 ms | PASS |
| 99th Percentile Latency | ~5.20 ms | 13.91 ms | — | OK |

---

### Supporting Sections (Dense Modules)

**Panel A: Statistical Rigor**
- Bootstrap F1 95% CI: [0.7593, 0.9195]
- Hypothesis p-value: 0.1501
- Statistical power at N_fraud=52: 24.8%
- Required N_fraud for 80% power: 325 (188,953 total transactions)

**Panel B: Methodology Guardrails**
- Temporal split: 60/20/20 chronological
- Resampling: SMOTE + RUS, train-only (1:5 ratio)
- Pruning threshold: 8ms 95th percentile latency
- Decision threshold: 0.7600

**Panel C: Feature Engineering**
- 72 total features × 4 families
- Temporal: cyclical hour, night/weekend flags, velocity
- Rolling: mean/std over windows 3/5/10, Z-score
- Interaction: PCA crosses (Amount×V1, V4, V7), squared
- Anomalies: expanding cumulative, global Z-score

**Panel D: Literature Audit**
- Ours: Strict chronological, no leakage → F1 0.8478 (realistic)
- 35+ audited papers: Global SMOTE/RUS → inflated F1 0.91-0.999
- Key insight: Temporal isolation = production generalization

---

### Visual Cues & Labels

Use corporate-memphis styling:
- Flat vector comparison blocks
- Vibrant purple/orange/teal fills
- Clean sans-serif typography
- PASS badges for metrics meeting targets
- NEAR TARGET for close-but-unmet goals
- Side-by-side metric bars showing baseline vs optimized values
- Highlight the precision improvement (86.67% → 97.50%) as the key business impact

---

### Aspect Ratio Mapping

landscape (16:9)
