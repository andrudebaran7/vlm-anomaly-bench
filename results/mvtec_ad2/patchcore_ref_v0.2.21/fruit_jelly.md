# MVTec AD 2 — patchcore_ref on fruit_jelly

Generated 2026-10-25 17:06 UTC by `scripts/run_mvtec_ad2.py`.

Mean ± std over **3 seed(s)** (0, 1, 2), each aggregated separately (protocol §6). The public test split; `test_private` has no ground truth here.

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| overexposed | 0.8089 ± 0.0077 | 0.6371 ± 0.0021 | 0.3840 ± 0.0039 |
| regular | 0.7600 ± 0.0133 | 0.6325 ± 0.0026 | 0.3709 ± 0.0034 |
| shift_1 | 0.7600 ± 0.0231 | 0.6372 ± 0.0025 | 0.3833 ± 0.0041 |
| underexposed | 0.7644 ± 0.0204 | 0.6284 ± 0.0026 | 0.3763 ± 0.0028 |

## Caveats recorded with the result

- **PatchCore is the full-shot anchor, not a zero-shot method.** It is the ceiling the zero-shot numbers are measured against, never a comparable entry.
- **One category is not a dataset result.** MVTec AD 2 has eight.
- **Protocol v0.2.21 configuration:** forward resizes with antialiasing, the same pre-processing the memory bank was built with. Numbers from before 2026-10-19 used a non-antialiased scoring path and are a different configuration; they are not comparable entries.
- The anchor passed its MVTec AD classic gate under this configuration (`results/reproduction/patchcore_mvtec_ad_v0.2.21.md`).
- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio preservation, and no MVTec AD 2 category is square. This is the pinned configuration the reproduction gate was run at (protocol §7); it is a stated property of the study, not an explanation produced afterwards.

## Provenance

- shards: `results/mvtec_ad2/patchcore_ref_v0.2.21/fruit_jelly/shards`
- seeds: `0, 1, 2`
- commit: `6f4b2a0`
- anomalib_version: `2.6.0`
- torch_version: `2.11.0+cu130`
