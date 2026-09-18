# Reproduction gate — patchcore_ref on mvtec_ad

Generated 2026-09-18 16:51 UTC by `scripts/reproduction_gate.py`.

## Verdict: **PASS**

Measured mean I-AUROC **98.02 ± 0.07** against published **99.0** (delta -0.98, tolerance ±1.0).

⚠️ **6 of 15 categories are outside ±1.0 on their own:** cable (-1.55), carpet (-1.08), pill (-1.83), tile (+1.10), toothbrush (-8.22), zipper (-1.12). The gate is on the mean, so this does not change the verdict — but it is not an even reproduction, and the cause belongs in the write-up before any frontier number is reported.

Source: arXiv:2106.08265 Table S1, row PatchCore-10

## Per seed

| Seed | Mean I-AUROC |
|---|---|
| seed0 | 97.94 |
| seed1 | 98.08 |
| seed2 | 98.02 |

## Per category

| Category | Measured | Published | Delta |
|---|---|---|---|
| bottle | 100.00 ± 0.00 | 100.0 | +0.00 |
| cable | 97.85 ± 0.42 | 99.4 | -1.55 ⚠️ |
| capsule | 98.74 ± 0.28 | 97.8 | +0.94 |
| carpet | 97.62 ± 0.16 | 98.7 | -1.08 ⚠️ |
| grid | 97.99 ± 0.08 | 97.9 | +0.09 |
| hazelnut | 100.00 ± 0.00 | 100.0 | +0.00 |
| leather | 100.00 ± 0.00 | 100.0 | +0.00 |
| metal_nut | 99.25 ± 0.07 | 100.0 | -0.75 |
| pill | 94.17 ± 0.30 | 96.0 | -1.83 ⚠️ |
| screw | 96.85 ± 0.42 | 97.0 | -0.15 |
| tile | 100.00 ± 0.00 | 98.9 | +1.10 ⚠️ |
| toothbrush | 91.48 ± 0.58 | 99.7 | -8.22 ⚠️ |
| transistor | 99.04 ± 0.23 | 100.0 | -0.96 |
| wood | 98.86 ± 0.09 | 99.0 | -0.14 |
| zipper | 98.38 ± 0.13 | 99.5 | -1.12 ⚠️ |

## Provenance

- results_root: `results/reproduction/mvtec_ad/shards`
- targets: `configs/reproduction/patchcore_ref.yaml`
- n_categories: `15`
- commit: `45462ea3886e356865672bd88c0b6416e95e8d4c`
- seed: `['0', '1', '2']`
- gpu: `Tesla T4`
- preprocess: `anomalib`
