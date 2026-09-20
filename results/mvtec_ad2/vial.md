# MVTec AD 2 — patchcore_ref on vial

**The first reportable MVTec AD 2 number of this study.** Protocol §2 blocked every AD 2 result
until PatchCore reproduced its published figures; its gate passed on 2026-09-18 (MVTec AD classic,
98.02 ± 0.07 against a published 99.0), which is what unblocked this.

Run 2026-09-21 on a Colab Tesla T4, anomalib 2.6.0, `preprocess=anomalib`, seeds 0/1/2, the
`test_public` split (140 images across 7 lighting conditions, 20 each).

**Transcribed by hand from the session console.** The runner printed these numbers but wrote no
file — that gap is fixed in the same commit as this report, so every later category writes its own.
The commit is recorded as a range rather than a point for the same reason: the run used the
`mean ± std` summary introduced in `7a03f14`, so it is that commit or later.

Mean ± std over **3 seeds**, each aggregated separately (protocol §6):

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| regular | 0.95 ± 0.01 | 0.85 ± 0.00 | 0.59 ± 0.01 |
| overexposed | 0.92 ± 0.01 | 0.87 ± 0.00 | 0.60 ± 0.01 |
| shift_1 | 0.92 ± 0.02 | 0.85 ± 0.00 | 0.57 ± 0.00 |
| shift_2 | 0.88 ± 0.02 | 0.86 ± 0.00 | 0.59 ± 0.01 |
| shift_3 | 0.92 ± 0.01 | 0.86 ± 0.00 | 0.60 ± 0.01 |
| shift_4 | 0.91 ± 0.01 | 0.85 ± 0.00 | 0.56 ± 0.01 |
| underexposed | 0.87 ± 0.05 | 0.75 ± 0.01 | 0.49 ± 0.00 |

Values are on 0–1 here, where the reproduction-gate reports use 0–100. **A ± of 0.00 means below
0.005, not identical**; the console printed two decimals and the transcription cannot recover more.
Later categories are written by the runner at four decimals.

## What it shows

- **The `regular` condition reproduces phase 2's single-seed 0.947** (2026-09-16), on a different
  run three seeds later. That consistency check was free and it passes.
- **Lighting shift degrades detection**: `regular` 0.95 down to `underexposed` 0.87, with
  AU-PRO@30% falling 0.85 → 0.75 and AU-PRO@5% 0.59 → 0.49. This is the axis MVTec AD 2 was built
  to measure, and the full-shot anchor is not immune to it.
- **`underexposed` is also the least stable across seeds** (±0.05 against ±0.01–0.02 everywhere
  else). The hardest condition is the one where the seed matters most, which is an argument for
  §6's three seeds rather than a curiosity.
- **AU-PRO@5% sits far below AU-PRO@30%** (0.49–0.60 against 0.75–0.87). Detection quality and
  localisation quality separate here, which is what §5.2 of the write-up exists to examine.

## Caveats recorded with the result

- **PatchCore is the full-shot anchor, not a zero-shot method.** It is the ceiling the zero-shot
  numbers are measured against, never a comparable entry.
- **One category is not a dataset result.** MVTec AD 2 has eight; this is the smallest.
- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio preservation,
  and Vial is 1400×1900. No MVTec AD 2 category is square. This is the pinned configuration the
  reproduction gate was passed at (protocol §7) — a stated property of the study, not an
  explanation produced afterwards.
- **Nothing here is compared to the dataset paper's "below 60% average AU-PRO" figure**, which is
  a dataset-level average over methods and categories under its own protocol. One category of a
  full-shot anchor is not that number.

## Provenance

- shards: `results/mvtec_ad2/vial/shards`, also on Drive at `MyDrive/mvtec_ad2_results/vial/`
- seeds: `0, 1, 2`
- coreset: 29797 indices at every seed — `floor(291 × 102.4)`, the 32×32 grid, confirming the
  `anomalib` transform reached all three runs
- fit: 2m19s / 2m20s / 2m20s, against the ~2m20s the cost model predicted
- commit: `7a03f14` or later
