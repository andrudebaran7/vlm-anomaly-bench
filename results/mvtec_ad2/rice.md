# MVTec AD 2 — patchcore_ref on rice

Run 2026-10-01, Colab Tesla T4, anomalib 2.6.0, `preprocess=anomalib`, seeds 0/1/2, `test_public`
(132 images over **6** lighting conditions, **22** each). Commit `67a3e69`.

The table below is the runner's own output. The sections after it were added by hand.

Mean ± std over **3 seeds**, each aggregated separately (protocol §6):

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| overexposed | 0.4952 ± 0.0660 | 0.2904 ± 0.0058 | 0.1182 ± 0.0084 |
| regular | 0.5397 ± 0.0275 | 0.3049 ± 0.0052 | 0.1384 ± 0.0037 |
| shift_1 | 0.5619 ± 0.0660 | 0.3361 ± 0.0083 | 0.1444 ± 0.0059 |
| shift_2 | 0.5524 ± 0.0530 | 0.4491 ± 0.0077 | 0.2338 ± 0.0087 |
| shift_3 | 0.4381 ± 0.0165 | 0.4171 ± 0.0045 | 0.2233 ± 0.0039 |
| underexposed | 0.5270 ± 0.0291 | 0.3240 ± 0.0116 | 0.1586 ± 0.0057 |

## The result, stated plainly

**Detection is at chance; localisation is real.** I-AUROC averages 0.519 over the six conditions
and ranges 0.438–0.562, straddling 0.50 with per-condition standard deviations up to 0.066.
Meanwhile `au_pro_030` averages 0.3536 — **2.4× the measured random baseline of 0.1487** — and
`au_pro_005` averages 0.1695, **6.8× its 0.0249 baseline**. The maps carry a usable signal; the
image-level score does not separate the classes.

**This category has the coarsest detection reading in the grid, and it matters here.**
`test_public` gives rice 42 normal and 90 anomalous images (Table 4) over six conditions, so each
condition has **7 normal and 15 anomalous** — 105 pairs, steps of 0.0095, and only seven normal
images to place. That is the fewest of the eight categories. The ±0.066 on two conditions is what
seven normals buys. "At chance" is the honest reading, but it is a low-resolution one: this
category could not have produced a confident number either way.

## The direction-shift pattern is inverted here, and that ends it as a general claim

On Wall Plugs, Can and Walnuts the three `shift_*` conditions scored *below* the three intensity
conditions on `au_pro_005`. On rice they score **above**, and not marginally:

| | intensity (regular / over / under) | shifts (1 / 2 / 3) |
|---|---|---|
| `au_pro_005` | 0.1384 / 0.1182 / 0.1586 → **0.1384** | 0.1444 / 0.2338 / 0.2233 → **0.2005** |
| `au_pro_030` | 0.3049 / 0.2904 / 0.3240 → **0.3064** | 0.3361 / 0.4491 / 0.4171 → **0.4008** |

`shift_2` and `shift_3` are rice's two *best* conditions on both pixel metrics, by a wide margin.

**The tally is now three for and three against** — Wall Plugs, Can and Walnuts in one direction;
Vial (worst condition is `underexposed`, an intensity change), Fruit Jelly (`shift_1` is its best)
and rice in the other. **The pattern does not survive seven categories and must not be written as
a finding.** A note in `docs/next-steps.md` dated 2026-09-30 predicted that a fourth reading would
make it writable; the fourth reading refuted it instead. Recorded both ways, which is the point of
having written the prediction down.

## What seven categories do show: the two metrics agree ACROSS categories

Per-category means, with each pixel metric expressed as a multiple of its measured random
baseline:

| category | i_auroc | au_pro_030 (×random) | au_pro_005 (×random) |
|---|---|---|---|
| Vial | 0.910 | 0.8414 (5.7×) | 0.5714 (22.9×) |
| Walnuts | 0.824 | 0.7595 (5.1×) | 0.4479 (18.0×) |
| Fruit Jelly | 0.783 | 0.6394 (4.3×) | 0.3865 (15.5×) |
| Rice | 0.519 | 0.3536 (2.4×) | 0.1695 (6.8×) |
| Sheet Metal | 0.510 | 0.2523 (1.7×) | 0.0677 (2.7×) |
| Wall Plugs | 0.447 | 0.4194 (2.8×) | 0.1189 (4.8×) |
| Can | 0.388 | 0.2984 (2.0×) | 0.0024 (0.1×) |

**Ordering the seven by I-AUROC and by `au_pro_005` gives the same sequence apart from one
adjacent transposition** (Sheet Metal and Wall Plugs swap). Where this anchor localises well it
detects well, and where it localises badly it detects badly.

**This corrects a reading the earlier reports were drifting towards.** Wall Plugs' and Can's
reports describe detection failing while localisation survives, and that is true *within* those
categories. It does not generalise into "the image-level score is the weak link": across
categories the two move together. The dissociation this study has evidence for is **between
lighting conditions inside a category**, not between the two metrics as such.

## Caveats

- **PatchCore is the full-shot anchor, not a zero-shot method.** At 0.519 mean I-AUROC it is not a
  usable detection ceiling here; its localisation, at 6.8× random, is still a meaningful one.
- **One category is not a dataset result.** MVTec AD 2 has eight; this is the seventh, and the
  grid stands at three usable anchors and four unusable.
- **Seven normal images per lighting condition.** See above — the coarsest per-condition detection
  reading of the grid.
- **The image-level score is anomalib's `pred_score`**, the model's own reduction of patch scores
  to a scalar (`patchcore_backend.py` line 297). It is not computed by this repository, so any
  claim about the reduction is a claim about the upstream implementation.
- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio preservation,
  and rice is 2448×2048 (1.20:1) — the same frame as Wall Plugs, which fails, and Walnuts, which
  is the best in the grid. The pinned configuration the reproduction gate was passed at (§7).

## Provenance

- shards: `results/mvtec_ad2/rice/shards`, and `MyDrive/mvtec_ad2_results/rice/`
- seeds: `0, 1, 2`
- coreset: **not captured**; it is **32051** by `floor(313 × 102.4)`, the relationship measured on
  2026-09-30. Fourth category running where the figure was not recorded — see `docs/next-steps.md`
  on why the adapter should stamp it rather than a human copying a progress bar.
- fit: **not captured**. The cost model predicts 2.7m per seed, ~8m for three.
- the summary needed `--summarise-only --max-bytes 12000000000`: 22 images × 2448×2048 is
  110,297,088 pooled pixels = **8.82 GB**, over the 6 GB pixel-metric guard
- upload: 6.29 GB to Drive, **~2 hours** on the author's connection. Recorded because it is the
  grid's real cost and the reason `fabric` (10 GB) is the last category.
- commit: `67a3e69`
