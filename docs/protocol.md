# Evaluation protocol (frozen before results)

The point of freezing this document is credibility: every choice below is made **before** seeing a
single result, so no metric or threshold can be quietly tuned to make one method look good.
Any later change gets a dated entry in the Changelog at the bottom.

## 1. Task definition

Image-level anomaly detection + pixel-level anomaly localisation, **zero-shot** (no target-category
training images) unless a method's protocol explicitly defines an auxiliary-training regime
(AnomalyCLIP, AdaCLIP: trained on auxiliary datasets that must NOT overlap with any test dataset —
overlap table required in the paper).

## 2. Datasets & splits

- **MVTec AD 2** — official test protocol, including the lighting-variation test sets. Report
  per-category and mean. *(TODO before M1: verify official split names and the evaluation-server
  submission rules from the official MVTec page; do not rely on memory.)*
- **Real-IAD Variety** — official protocol. *(TODO before M1: confirm access terms and exact subset
  definitions once the application is approved.)*
- **VisA + MVTec AD (classic)** — used only to validate our re-implementations against published
  numbers (tolerance ±1.0 I-AUROC). If we can't reproduce a method's published VisA/MVTec numbers,
  its frontier results are flagged as such in every table.

## 3. Methods & implementations

Priority order: (1) official code, pinned commit; (2) anomalib implementation, pinned version;
(3) re-implementation (last resort, flagged). Prompts for CLIP-based methods are taken verbatim
from the original papers. The MLLM baseline uses one fixed structured prompt + one fixed scoring
rubric for all datasets, published in `configs/methods/mllm_qwen.yaml` — no per-category prompt
engineering.

## 4. Metrics

- Image level: I-AUROC, I-AP, I-F1max.
- Pixel level: P-AUROC, AU-PRO@0.3, AU-PRO@0.05, and the MVTec AD 2 official metric set.
- Efficiency: median and p95 latency per image at batch size 1 on the fixed hardware below,
  parameter count, peak VRAM. API-based baselines report tokens + cost instead of VRAM.

## 5. Hardware & software

One fixed environment for all latency numbers (single GPU machine or a rented fixed instance —
documented exactly in `results/ENVIRONMENT.md` with driver/library versions). Accuracy metrics may
be computed anywhere; latency only on the reference machine.

## 6. Statistical hygiene

Three seeds where any stochasticity exists; report mean ± std. No cherry-picking categories:
every table reports all categories or states explicitly why one is excluded (e.g., official
protocol exclusion). Failed runs are reported as failures, not silently dropped.

## 7. What we will NOT do

- No per-dataset prompt tuning after seeing test results.
- No "best of N runs".
- No mixing of our PatchCore anchor numbers with published numbers in the same column without marking.

## Changelog

- 2026-07-11 — v0.1 initial frozen draft.
