# MVTec AD 2 — patchcore_ref on rice

Generated 2026-10-25 22:16 UTC by `scripts/run_mvtec_ad2.py`.

Mean ± std over **3 seed(s)** (0, 1, 2), each aggregated separately (protocol §6). The public test split; `test_private` has no ground truth here.

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| overexposed | 0.4603 ± 0.0334 | 0.3861 ± 0.0062 | 0.1391 ± 0.0136 |
| regular | 0.5429 ± 0.0190 | 0.4038 ± 0.0073 | 0.1423 ± 0.0128 |
| shift_1 | 0.5460 ± 0.0306 | 0.4658 ± 0.0080 | 0.1364 ± 0.0117 |
| shift_2 | 0.5524 ± 0.0330 | 0.4640 ± 0.0064 | 0.2379 ± 0.0071 |
| shift_3 | 0.4984 ± 0.0291 | 0.4864 ± 0.0072 | 0.2502 ± 0.0083 |
| underexposed | 0.6190 ± 0.0756 | 0.3928 ± 0.0092 | 0.1312 ± 0.0125 |

## Caveats recorded with the result

- **PatchCore is the full-shot anchor, not a zero-shot method.** It is the ceiling the zero-shot numbers are measured against, never a comparable entry.
- **One category is not a dataset result.** MVTec AD 2 has eight.
- **Protocol v0.2.21 configuration:** forward resizes with antialiasing, the same pre-processing the memory bank was built with. Numbers from before 2026-10-19 used a non-antialiased scoring path and are a different configuration; they are not comparable entries.
- The anchor passed its MVTec AD classic gate under this configuration (`results/reproduction/patchcore_mvtec_ad_v0.2.21.md`).
- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio preservation, and no MVTec AD 2 category is square. This is the pinned configuration the reproduction gate was run at (protocol §7); it is a stated property of the study, not an explanation produced afterwards.

## Provenance

- shards: `results/mvtec_ad2/patchcore_ref_v0.2.21/rice/shards`
- seeds: `0, 1, 2`
- commit: `6f4b2a0`
- anomalib_version: `2.6.0`
- torch_version: `2.11.0+cu130`
