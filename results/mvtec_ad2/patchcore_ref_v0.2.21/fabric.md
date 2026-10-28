# MVTec AD 2 — patchcore_ref on fabric

Generated 2026-10-28 14:57 UTC by `scripts/run_mvtec_ad2.py`.

Mean ± std over **3 seed(s)** (0, 1, 2), each aggregated separately (protocol §6). The public test split; `test_private` has no ground truth here.

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| overexposed | 0.7374 ± 0.0273 | 0.2194 ± 0.0380 | 0.0074 ± 0.0012 |
| regular | 0.7394 ± 0.0219 | 0.2161 ± 0.0458 | 0.0063 ± 0.0008 |
| shift_1 | 0.7313 ± 0.0229 | 0.2714 ± 0.0444 | 0.0060 ± 0.0001 |
| shift_2 | 0.6707 ± 0.0394 | 0.2975 ± 0.0218 | 0.0940 ± 0.0101 |
| shift_3 | 0.7152 ± 0.0061 | 0.3114 ± 0.0334 | 0.0484 ± 0.0183 |
| underexposed | 0.7414 ± 0.0286 | 0.2065 ± 0.0437 | 0.0057 ± 0.0003 |

## Caveats recorded with the result

- **PatchCore is the full-shot anchor, not a zero-shot method.** It is the ceiling the zero-shot numbers are measured against, never a comparable entry.
- **One category is not a dataset result.** MVTec AD 2 has eight.
- **Protocol v0.2.21 configuration:** forward resizes with antialiasing, the same pre-processing the memory bank was built with. Numbers from before 2026-10-19 used a non-antialiased scoring path and are a different configuration; they are not comparable entries.
- The anchor passed its MVTec AD classic gate under this configuration (`results/reproduction/patchcore_mvtec_ad_v0.2.21.md`).
- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio preservation, and no MVTec AD 2 category is square. This is the pinned configuration the reproduction gate was run at (protocol §7); it is a stated property of the study, not an explanation produced afterwards.

## Provenance

- shards: `results/mvtec_ad2/patchcore_ref_v0.2.21/fabric/shards`
- seeds: `0, 1, 2`
- commit: `6f4b2a0`
- anomalib_version: `2.6.0`
- torch_version: `2.11.0+cu130`
