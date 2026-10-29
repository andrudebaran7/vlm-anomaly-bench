# Reproduction gate — winclip on mvtec_ad

Generated 2026-10-29 15:15 UTC by `scripts/reproduction_gate.py`.

## Verdict: **FAIL**

Measured mean I-AUROC **90.37** against published **91.8** (delta -1.43, tolerance ±1.0).

⚠️ **7 of 15 categories are outside ±1.0 on their own:** cable (-1.55), capsule (-4.61), hazelnut (-1.15), pill (+2.11), screw (-11.13), toothbrush (-1.94), wood (-1.77). The gate is on the mean, so this does not change the verdict — but it is not an even reproduction, and the cause belongs in the write-up before any frontier number is reported.

Source: arXiv:2303.14814 Table 1, 0-shot, WinCLIP (ours), MVTec-AD AUROC

## Per category

| Category | Measured | Published | Delta |
|---|---|---|---|
| bottle | 98.65 | 99.2 | -0.55 |
| cable | 84.95 | 86.5 | -1.55 ⚠️ |
| capsule | 68.29 | 72.9 | -4.61 ⚠️ |
| carpet | 99.32 | 100.0 | -0.68 |
| grid | 99.16 | 98.8 | +0.36 |
| hazelnut | 92.75 | 93.9 | -1.15 ⚠️ |
| leather | 100.00 | 100.0 | +0.00 |
| metal_nut | 96.29 | 97.1 | -0.81 |
| pill | 81.21 | 79.1 | +2.11 ⚠️ |
| screw | 72.17 | 83.3 | -11.13 ⚠️ |
| tile | 99.93 | 100.0 | -0.07 |
| toothbrush | 85.56 | 87.5 | -1.94 ⚠️ |
| transistor | 88.83 | 88.0 | +0.83 |
| wood | 97.63 | 99.4 | -1.77 ⚠️ |
| zipper | 90.84 | 91.5 | -0.66 |

## Provenance

- results_root: `results/reproduction/winclip/mvtec_ad/shards`
- targets: `configs/reproduction/winclip.yaml`
- n_categories: `15`
- commit: `6f4b2a0809f0d0a5fa82e30380192f2157073e58`
- seed: `0`
- gpu: `Tesla T4`
- preprocess: `None`
- resize_antialias: `None`
- anomalib_version: `2.6.0`
- torch_version: `2.11.0+cu130`
