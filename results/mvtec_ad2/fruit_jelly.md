# MVTec AD 2 — patchcore_ref on fruit_jelly

Run 2026-09-29, Colab Tesla T4, anomalib 2.6.0, `preprocess=anomalib`, seeds 0/1/2, `test_public`
(80 images over **4** lighting conditions, 20 each). Commit `61704b4`.

The table below is the runner's own output. The sections after it were added by hand.

Mean ± std over **3 seeds**, each aggregated separately (protocol §6):

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| overexposed | 0.8533 ± 0.0133 | 0.6453 ± 0.0016 | 0.3886 ± 0.0042 |
| regular | 0.7911 ± 0.0204 | 0.6383 ± 0.0011 | 0.3820 ± 0.0035 |
| shift_1 | 0.6844 ± 0.0204 | 0.6445 ± 0.0010 | 0.3925 ± 0.0033 |
| underexposed | 0.8044 ± 0.0278 | 0.6295 ± 0.0016 | 0.3829 ± 0.0029 |

## The result, stated plainly

**Detection moves with the lighting; localisation does not.** I-AUROC spans 0.684 to 0.853 across
the four conditions — a range of 0.169. AU-PRO@30% spans 0.6295 to 0.6453, a range of **0.016**,
with per-seed standard deviations of 0.001–0.002. The same model, on the same images, ranks whole
images very differently under different lighting while segmenting the anomalous regions in them
about equally well.

**`shift_1` is the sharpest case and it runs the two metrics in opposite directions.** It has the
worst detection of the four (0.6844) and the second-best localisation (0.6445, 0.001 below the
best). Whatever the shift does, it is not destroying the anomaly signal in the map: it is moving
the per-image score distributions of normal and anomalous images together.

**`regular` is not the best condition, on either metric.** The training images are captured under
`regular` only, so the naive expectation is that it is the ceiling and every other condition falls
from it. It is not: `overexposed` beats it on I-AUROC by 0.062 and on AU-PRO by 0.007. This is the
second category where `regular` is not the maximum — see the caveat on granularity below before
reading much into the I-AUROC part of it.

## What this is NOT

**It is not the Vial pattern.** On Vial both metrics degraded together and in the same direction:
`underexposed` took I-AUROC from 0.95 to 0.87 and AU-PRO@30% from 0.85 to 0.75. Fruit Jelly's
AU-PRO does not move at all. So "lighting shift degrades localisation" is **not** a finding of this
study so far; what the two categories have in common is only that lighting shift degrades
*detection*. Any statement about localisation under lighting has to be made per category, on the
evidence of two.

**It is not comparable to the dataset paper's "below 60% average AU-PRO".** That is a
dataset-level average over eight categories under the paper's own protocol, not a per-category and
certainly not a per-lighting-condition figure. Nothing here is scored against it.

## Caveats

- **PatchCore is the full-shot anchor, not a zero-shot method.** It is the ceiling the zero-shot
  numbers are measured against, never a comparable entry.
- **One category is not a dataset result.** MVTec AD 2 has eight; this is the third.
- **Per-condition I-AUROC is coarse.** `test_public` gives this category 20 normal and 60
  anomalous images (Table 4) over four conditions, so each condition has **5 normal and 15
  anomalous** — 75 pairs, and I-AUROC can only take values k/75, in steps of 0.0133. A ± of
  0.0133 is therefore exactly one step: the difference between `regular` and `overexposed` is
  4.6 steps and is probably real; smaller differences are not readable. AU-PRO is not subject to
  this — it is computed over pixels, which is also why its ± is an order of magnitude smaller.
- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio preservation,
  and Fruit Jelly is 2100×1520 (1.38:1). This is the pinned configuration the reproduction gate
  was passed at (protocol §7); it is a stated property of the study, not an explanation produced
  afterwards. The compression here is mild next to Sheet Metal's 4:1.

## The shards were computed twice, and that is this category's most useful result

This category ran first on **2026-09-20 at commit `a7bc7bd`**, in the same session as Sheet Metal,
and no report was written — the shards sat on Drive and the result existed nowhere else. The
2026-09-29 session found them, restored them, skipped all three seeds and wrote a report stamping
the *current* commit onto numbers computed nine days earlier at a different one. Those shards were
deleted and the category re-run from scratch.

**All four I-AUROC values reproduced to four decimals**, on a different VM, nine days apart:

| lighting | 2026-09-20, `a7bc7bd` | 2026-09-29, `61704b4` |
|---|---|---|
| overexposed | 0.8533 ± 0.0133 | 0.8533 ± 0.0133 |
| regular | 0.7911 ± 0.0204 | 0.7911 ± 0.0204 |
| shift_1 | 0.6844 ± 0.0204 | 0.6844 ± 0.0204 |
| underexposed | 0.8044 ± 0.0278 | 0.8044 ± 0.0278 |

There is no code difference between the two commits — `git diff a7bc7bd..61704b4 -- src scripts
configs` is empty — so this is a clean test of the seeding, not of the code. **It is the first
evidence in this study that `PatchCoreBackend(seed=n)` reproduces across sessions and machines.**
Both runs used a Tesla T4, so it does not establish anything across GPU models. The pixel metrics
have no such check: the 2026-09-20 run's maps died with its VM, which is why the category was
re-run rather than reported from the restored shards.

## Provenance

- shards: `results/mvtec_ad2/fruit_jelly/shards`, and `MyDrive/mvtec_ad2_results/fruit_jelly/`
- seeds: `0, 1, 2`; coreset **26930** indices each, at every seed
- fit: 1m51s at every seed, against the 1.9m the cost model predicted
- the summary needed no `--max-bytes`: 20 images × 2100×1520 is 63,840,000 pooled pixels =
  5.11 GB, under the 6 GB pixel-metric guard, as projected
- commit: `61704b4`
