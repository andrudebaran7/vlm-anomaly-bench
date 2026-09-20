# Reproduction secondary check — patchcore_ref on visa

Generated 2026-09-20 00:01 UTC by `scripts/reproduction_gate.py`.

## Secondary check — reported only, **does not gate**

Measured mean I-AUROC **86.26** against published **92.4** (delta -6.14).

Protocol §2 v0.2.12: this number cannot fail the method. Caveat recorded with the target:

> PatchCore's own paper predates VisA and reports no VisA number, so this comes from a third party. That source states none of its PatchCore hyperparameters (no resolution, crop or coreset ratio), and its MVTec-AD control in the same row is 99.8 — above every single-model number in PatchCore's own paper (99.0-99.1) and above its 99.6 ensemble. A miss here does not distinguish a wrong implementation from a different setup, which is why it does not gate.

Source: arXiv:2207.14315 (VisA dataset paper) Table 6, row 'Sup. pre-train', VisA 1-class

## Per category

No per-category breakdown is published for this target, so the measured values stand alone.

| Category | Measured |
|---|---|
| candle | 91.30 |
| capsules | 74.82 |
| cashew | 95.56 |
| chewinggum | 98.24 |
| fryum | 86.26 |
| macaroni1 | 81.85 |
| macaroni2 | 68.83 |
| pcb1 | 83.21 |
| pcb2 | 89.68 |
| pcb3 | 91.32 |
| pcb4 | 96.52 |
| pipe_fryum | 77.54 |

## Provenance

- results_root: `results/reproduction/visa/shards`
- targets: `configs/reproduction/patchcore_ref.yaml`
- n_categories: `12`
- commit: `8f043c7dcbb88a6161b10235edf27621303681d8`
- seed: `0`
- gpu: `Tesla T4`
- preprocess: `anomalib`
