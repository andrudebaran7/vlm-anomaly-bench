# MVTec AD 2 — patchcore_ref on can

Generated 2026-10-25 19:21 UTC by `scripts/run_mvtec_ad2.py`.

Mean ± std over **3 seed(s)** (0, 1, 2), each aggregated separately (protocol §6). The public test split; `test_private` has no ground truth here.

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| overexposed | 0.4519 ± 0.0370 | 0.2841 ± 0.0018 | 0.0401 ± 0.0016 |
| regular | 0.4519 ± 0.0285 | 0.2903 ± 0.0081 | 0.0372 ± 0.0014 |
| shift_1 | 0.3704 ± 0.0463 | 0.2478 ± 0.0099 | 0.0045 ± 0.0047 |
| shift_2 | 0.4444 ± 0.0147 | 0.1913 ± 0.0061 | 0.0021 ± 0.0024 |
| shift_3 | 0.4759 ± 0.0085 | 0.1868 ± 0.0050 | 0.0000 ± 0.0000 |
| underexposed | 0.5148 ± 0.0672 | 0.2205 ± 0.0048 | 0.0082 ± 0.0017 |

## Caveats recorded with the result

- **PatchCore is the full-shot anchor, not a zero-shot method.** It is the ceiling the zero-shot numbers are measured against, never a comparable entry.
- **One category is not a dataset result.** MVTec AD 2 has eight.
- **Protocol v0.2.21 configuration:** forward resizes with antialiasing, the same pre-processing the memory bank was built with. Numbers from before 2026-10-19 used a non-antialiased scoring path and are a different configuration; they are not comparable entries.
- The anchor passed its MVTec AD classic gate under this configuration (`results/reproduction/patchcore_mvtec_ad_v0.2.21.md`).
- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio preservation, and no MVTec AD 2 category is square. This is the pinned configuration the reproduction gate was run at (protocol §7); it is a stated property of the study, not an explanation produced afterwards.

## Provenance

- shards: `results/mvtec_ad2/patchcore_ref_v0.2.21/can/shards`
- seeds: `0, 1, 2`
- commit: `6f4b2a0`
- anomalib_version: `2.6.0`
- torch_version: `2.11.0+cu130`
