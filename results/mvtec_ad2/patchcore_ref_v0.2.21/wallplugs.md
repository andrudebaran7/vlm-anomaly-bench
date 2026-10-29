# MVTec AD 2 — patchcore_ref on wallplugs

Generated 2026-10-25 20:30 UTC by `scripts/run_mvtec_ad2.py`.

Mean ± std over **3 seed(s)** (0, 1, 2), each aggregated separately (protocol §6). The public test split; `test_private` has no ground truth here.

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| overexposed | 0.4978 ± 0.0454 | 0.3674 ± 0.0118 | 0.1595 ± 0.0045 |
| regular | 0.4956 ± 0.0379 | 0.3610 ± 0.0129 | 0.1577 ± 0.0031 |
| shift_1 | 0.5578 ± 0.0582 | 0.4397 ± 0.0283 | 0.0963 ± 0.0282 |
| shift_2 | 0.3422 ± 0.0139 | 0.2173 ± 0.0050 | 0.0243 ± 0.0019 |
| shift_3 | 0.5267 ± 0.0371 | 0.3633 ± 0.0288 | 0.0904 ± 0.0073 |
| underexposed | 0.4911 ± 0.0367 | 0.3573 ± 0.0132 | 0.1545 ± 0.0026 |

## Caveats recorded with the result

- **PatchCore is the full-shot anchor, not a zero-shot method.** It is the ceiling the zero-shot numbers are measured against, never a comparable entry.
- **One category is not a dataset result.** MVTec AD 2 has eight.
- **Protocol v0.2.21 configuration:** forward resizes with antialiasing, the same pre-processing the memory bank was built with. Numbers from before 2026-10-19 used a non-antialiased scoring path and are a different configuration; they are not comparable entries.
- The anchor passed its MVTec AD classic gate under this configuration (`results/reproduction/patchcore_mvtec_ad_v0.2.21.md`).
- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio preservation, and no MVTec AD 2 category is square. This is the pinned configuration the reproduction gate was run at (protocol §7); it is a stated property of the study, not an explanation produced afterwards.

## Provenance

- shards: `results/mvtec_ad2/patchcore_ref_v0.2.21/wallplugs/shards`
- seeds: `0, 1, 2`
- commit: `6f4b2a0`
- anomalib_version: `2.6.0`
- torch_version: `2.11.0+cu130`
