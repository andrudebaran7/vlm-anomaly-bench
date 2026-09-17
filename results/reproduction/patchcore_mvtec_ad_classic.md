# Reproduction gate — patchcore_ref on mvtec_ad

Generated 2026-09-17 14:59 UTC by `scripts/reproduction_gate.py`.

## Verdict: **FAIL**

Measured mean I-AUROC **93.41** against published **99.0** (delta -5.59, tolerance ±1.0).

⚠️ **10 of 15 categories are outside ±1.0 on their own:** cable (-2.12), capsule (-39.60), carpet (-1.03), metal_nut (-2.00), pill (-2.76), screw (-16.72), tile (+1.10), toothbrush (-15.26), transistor (-3.00), zipper (-1.02). The gate is on the mean, so this does not change the verdict — but it is not an even reproduction, and the cause belongs in the write-up before any frontier number is reported.

Source: arXiv:2106.08265 Table S1, row PatchCore-10

## Per category

| Category | Measured | Published | Delta |
|---|---|---|---|
| bottle | 100.00 | 100.0 | +0.00 |
| cable | 97.28 | 99.4 | -2.12 ⚠️ |
| capsule | 58.20 | 97.8 | -39.60 ⚠️ |
| carpet | 97.67 | 98.7 | -1.03 ⚠️ |
| grid | 97.91 | 97.9 | +0.01 |
| hazelnut | 100.00 | 100.0 | +0.00 |
| leather | 100.00 | 100.0 | +0.00 |
| metal_nut | 98.00 | 100.0 | -2.00 ⚠️ |
| pill | 93.24 | 96.0 | -2.76 ⚠️ |
| screw | 80.28 | 97.0 | -16.72 ⚠️ |
| tile | 100.00 | 98.9 | +1.10 ⚠️ |
| toothbrush | 84.44 | 99.7 | -15.26 ⚠️ |
| transistor | 97.00 | 100.0 | -3.00 ⚠️ |
| wood | 98.68 | 99.0 | -0.32 |
| zipper | 98.48 | 99.5 | -1.02 ⚠️ |

## Provenance

- results_root: `results/reproduction/mvtec_ad_classic/shards`
- targets: `configs/reproduction/patchcore_ref.yaml`
- n_categories: `15`
- commit: `51c6404fc5b472f9445ba7be382c003503a8e30b`
- seed: `0`
- gpu: `Tesla T4`
- preprocess: `classic`
