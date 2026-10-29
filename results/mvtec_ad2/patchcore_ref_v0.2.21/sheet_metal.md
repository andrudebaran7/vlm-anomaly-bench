# MVTec AD 2 — patchcore_ref on sheet_metal

Generated 2026-10-25 16:38 UTC by `scripts/run_mvtec_ad2.py`.

Mean ± std over **3 seed(s)** (0, 1, 2), each aggregated separately (protocol §6). The public test split; `test_private` has no ground truth here.

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| overexposed | 0.8222 ± 0.0509 | 0.2344 ± 0.0023 | 0.0811 ± 0.0010 |
| regular | 0.8222 ± 0.0255 | 0.2433 ± 0.0069 | 0.0833 ± 0.0010 |
| shift_1 | 0.7500 ± 0.0667 | 0.2081 ± 0.0012 | 0.0513 ± 0.0051 |
| shift_2 | 0.7667 ± 0.0928 | 0.2057 ± 0.0022 | 0.0474 ± 0.0044 |
| shift_3 | 0.7444 ± 0.0509 | 0.2209 ± 0.0010 | 0.0595 ± 0.0061 |
| underexposed | 0.8778 ± 0.0481 | 0.2527 ± 0.0100 | 0.0853 ± 0.0017 |

## Caveats recorded with the result

- **PatchCore is the full-shot anchor, not a zero-shot method.** It is the ceiling the zero-shot numbers are measured against, never a comparable entry.
- **One category is not a dataset result.** MVTec AD 2 has eight.
- **Protocol v0.2.21 configuration:** forward resizes with antialiasing, the same pre-processing the memory bank was built with. Numbers from before 2026-10-19 used a non-antialiased scoring path and are a different configuration; they are not comparable entries.
- The anchor passed its MVTec AD classic gate under this configuration (`results/reproduction/patchcore_mvtec_ad_v0.2.21.md`).
- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio preservation, and no MVTec AD 2 category is square. This is the pinned configuration the reproduction gate was run at (protocol §7); it is a stated property of the study, not an explanation produced afterwards.

## Provenance

- shards: `results/mvtec_ad2/patchcore_ref_v0.2.21/sheet_metal/shards`
- seeds: `0, 1, 2`
- commit: `6f4b2a0`
- anomalib_version: `2.6.0`
- torch_version: `2.11.0+cu130`
