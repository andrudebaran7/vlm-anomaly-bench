# MVTec AD 2 — patchcore_ref on vial

Generated 2026-10-25 20:02 UTC by `scripts/run_mvtec_ad2.py`.

Mean ± std over **3 seed(s)** (0, 1, 2), each aggregated separately (protocol §6). The public test split; `test_private` has no ground truth here.

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| overexposed | 0.8711 ± 0.0204 | 0.8763 ± 0.0034 | 0.6019 ± 0.0014 |
| regular | 0.8800 ± 0.0353 | 0.9033 ± 0.0004 | 0.6067 ± 0.0013 |
| shift_1 | 0.9378 ± 0.0154 | 0.9025 ± 0.0006 | 0.5992 ± 0.0013 |
| shift_2 | 0.8800 ± 0.0231 | 0.9060 ± 0.0004 | 0.6086 ± 0.0009 |
| shift_3 | 0.8667 ± 0.0231 | 0.9063 ± 0.0005 | 0.6102 ± 0.0015 |
| shift_4 | 0.9289 ± 0.0077 | 0.9002 ± 0.0005 | 0.5866 ± 0.0006 |
| underexposed | 0.8533 ± 0.0481 | 0.6479 ± 0.0077 | 0.3591 ± 0.0108 |

## Caveats recorded with the result

- **PatchCore is the full-shot anchor, not a zero-shot method.** It is the ceiling the zero-shot numbers are measured against, never a comparable entry.
- **One category is not a dataset result.** MVTec AD 2 has eight.
- **Protocol v0.2.21 configuration:** forward resizes with antialiasing, the same pre-processing the memory bank was built with. Numbers from before 2026-10-19 used a non-antialiased scoring path and are a different configuration; they are not comparable entries.
- The anchor passed its MVTec AD classic gate under this configuration (`results/reproduction/patchcore_mvtec_ad_v0.2.21.md`).
- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio preservation, and no MVTec AD 2 category is square. This is the pinned configuration the reproduction gate was run at (protocol §7); it is a stated property of the study, not an explanation produced afterwards.

## Provenance

- shards: `results/mvtec_ad2/patchcore_ref_v0.2.21/vial/shards`
- seeds: `0, 1, 2`
- commit: `6f4b2a0`
- anomalib_version: `2.6.0`
- torch_version: `2.11.0+cu130`
