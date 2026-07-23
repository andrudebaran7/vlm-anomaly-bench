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

- **MVTec AD 2** — 8 scenarios, >8,000 high-resolution images. Defect-free train and validation
  sets; the test set is split into a public part with pixel-precise ground truth and a private part
  whose ground truth is withheld. Local metrics are computed on the public part only. The private
  part is scored solely through the official evaluation server (benchmark.mvtec.com).
  Licensed CC BY-NC-SA 4.0. Split names verified 2026-07-23 from the official download page:
  `train`, `validation`, `test_public`, `test_private`, `test_private_mixed`.

  `test_public` carries images from **every lighting condition** with pixel ground truth, so the
  lighting-robustness analysis is computed locally. `test_private` (regular lighting) and
  `test_private_mixed` (the same scenes under varied lighting) are a controlled paired
  comparison whose ground truth is withheld; they are reported only if server access is
  obtained. Their absence narrows the lighting result, it does not remove it.
- **Real-IAD Variety** — out of scope for this study. The dataset is publicly available (no
  application required), but its scale (198,950 images; the parent Real-IAD release is 622 GB, 53 GB
  for the 1024px variant) is incompatible with the Colab compute budget. Recorded as a stated
  limitation, not an omission.
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
- The official MVTec AD 2 metric set is adopted as the evaluation server defines it. *(Exact metric
  names and definitions are read from the server's submission documentation at download time.)*

## 5. Hardware & software

Accuracy metrics are computed on Google Colab (free tier, single T4, fp16). The assigned GPU varies
between sessions, so **no latency number is ever reported from a Colab session**. The efficiency
table is measured in M5 in a single session on one rented fixed instance, using the official MVTec
AD 2 runtime and memory-footprint utilities, documented in `results/ENVIRONMENT.md` with driver and
library versions. Every accuracy row records the GPU Colab assigned, so runs stay auditable despite
varying hardware.

## 6. Statistical hygiene

Three seeds where any stochasticity exists; report mean ± std. No cherry-picking categories:
every table reports all categories or states explicitly why one is excluded (e.g., official
protocol exclusion). Failed runs are reported as failures, not silently dropped.

## 7. What we will NOT do

- No per-dataset prompt tuning after seeing test results.
- No "best of N runs".
- No mixing of our PatchCore anchor numbers with published numbers in the same column without marking.
- **No iteration against the evaluation server.** One submission per method, final, made only after
  that method's configuration is frozen. The private test split has hidden ground truth; repeated
  submissions with selection of the best outcome is the same "best of N" forbidden above. Submission
  date and returned scores are recorded in `results/`.

## Changelog

- 2026-07-11 — v0.1 initial frozen draft.
- 2026-07-22 — v0.2. Scope corrected after verifying dataset facts that v0.1 assumed:
  Real-IAD requires no application and is 622 GB (removed from scope); MVTec AD 2 private test
  ground truth is evaluation-server only (§2, §4); Colab has no fixed hardware (§5); added the
  one-submission-per-method rule (§7). MLLM baseline pinned to Qwen2.5-VL-3B-Instruct to fit 16 GB.
  Rationale: docs/superpowers/specs/2026-07-22-scope-v0.2-colab-design.md
- 2026-07-23 — v0.2.1. Verified against the official download page, resolving two of the three
  items §2 left open: the five split directory names, and the 30.4 GB download size (available
  per category, largest 10 GB). Records that `test_public` spans every lighting condition with
  pixel ground truth, so the lighting-shift analysis does not depend on evaluation-server
  access. Official metric names remain unverified pending the server's submission
  documentation. No evaluation rule changed.
