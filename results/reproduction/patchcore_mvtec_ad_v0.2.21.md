# Reproduction gate — patchcore_ref on mvtec_ad

Generated 2026-10-21 01:59 UTC by `scripts/reproduction_gate.py`.

## Verdict: **PASS**

Measured mean I-AUROC **98.22 ± 0.03** against published **99.0** (delta -0.78, tolerance ±1.0).

⚠️ **5 of 15 categories are outside ±1.0 on their own:** capsule (+1.44), pill (-1.34), tile (+1.10), toothbrush (-8.31), zipper (-1.81). The gate is on the mean, so this does not change the verdict — but it is not an even reproduction, and the cause belongs in the write-up before any frontier number is reported.

Source: arXiv:2106.08265 Table S1, row PatchCore-10

## Per seed

| Seed | Mean I-AUROC |
|---|---|
| seed0 | 98.23 |
| seed1 | 98.18 |
| seed2 | 98.24 |

## Per category

| Category | Measured | Published | Delta |
|---|---|---|---|
| bottle | 100.00 ± 0.00 | 100.0 | +0.00 |
| cable | 98.71 ± 0.07 | 99.4 | -0.69 |
| capsule | 99.24 ± 0.11 | 97.8 | +1.44 ⚠️ |
| carpet | 98.77 ± 0.02 | 98.7 | +0.07 |
| grid | 98.86 ± 0.17 | 97.9 | +0.96 |
| hazelnut | 100.00 ± 0.00 | 100.0 | +0.00 |
| leather | 100.00 ± 0.00 | 100.0 | +0.00 |
| metal_nut | 99.77 ± 0.06 | 100.0 | -0.23 |
| pill | 94.66 ± 0.24 | 96.0 | -1.34 ⚠️ |
| screw | 96.15 ± 0.55 | 97.0 | -0.85 |
| tile | 100.00 ± 0.00 | 98.9 | +1.10 ⚠️ |
| toothbrush | 91.39 ± 0.28 | 99.7 | -8.31 ⚠️ |
| transistor | 99.43 ± 0.21 | 100.0 | -0.57 |
| wood | 98.57 ± 0.05 | 99.0 | -0.43 |
| zipper | 97.69 ± 0.11 | 99.5 | -1.81 ⚠️ |

## Provenance

- results_root: `results/reproduction/patchcore_ref_v0.2.21/mvtec_ad/shards`
- targets: `configs/reproduction/patchcore_ref.yaml`
- n_categories: `15`
- commit: `a15d3b6a1bc7a65ce386d2f12f070d9b6832ec1a`
- seed: `['0', '1', '2']`
- gpu: `Tesla T4`
- preprocess: `anomalib`
- resize_antialias: `True`
- anomalib_version: `2.6.0`
- torch_version: `2.11.0+cu130`
