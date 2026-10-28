# MVTec AD 2 — patchcore_ref on walnuts

Generated 2026-10-28 12:48 UTC by `scripts/run_mvtec_ad2.py`.

Mean ± std over **3 seed(s)** (0, 1, 2), each aggregated separately (protocol §6). The public test split; `test_private` has no ground truth here.

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| overexposed | 0.8533 ± 0.0200 | 0.7806 ± 0.0005 | 0.4700 ± 0.0077 |
| regular | 0.8556 ± 0.0168 | 0.7819 ± 0.0011 | 0.4695 ± 0.0086 |
| shift_1 | 0.9356 ± 0.0168 | 0.6585 ± 0.0016 | 0.3302 ± 0.0046 |
| shift_2 | 0.8244 ± 0.0473 | 0.7756 ± 0.0021 | 0.4492 ± 0.0108 |
| shift_3 | 0.8778 ± 0.0102 | 0.6980 ± 0.0040 | 0.3478 ± 0.0045 |
| underexposed | 0.8200 ± 0.0306 | 0.7821 ± 0.0027 | 0.4667 ± 0.0077 |

## Caveats recorded with the result

- **PatchCore is the full-shot anchor, not a zero-shot method.** It is the ceiling the zero-shot numbers are measured against, never a comparable entry.
- **One category is not a dataset result.** MVTec AD 2 has eight.
- **Protocol v0.2.21 configuration:** forward resizes with antialiasing, the same pre-processing the memory bank was built with. Numbers from before 2026-10-19 used a non-antialiased scoring path and are a different configuration; they are not comparable entries.
- The anchor passed its MVTec AD classic gate under this configuration (`results/reproduction/patchcore_mvtec_ad_v0.2.21.md`).
- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio preservation, and no MVTec AD 2 category is square. This is the pinned configuration the reproduction gate was run at (protocol §7); it is a stated property of the study, not an explanation produced afterwards.

## Provenance

- shards: `results/mvtec_ad2/patchcore_ref_v0.2.21/walnuts/shards`
- seeds: `0, 1, 2`
- commit: `6f4b2a0`
- anomalib_version: `2.6.0`
- torch_version: `2.11.0+cu130`
