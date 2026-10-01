# MVTec AD 2 — patchcore_ref on walnuts

Run 2026-09-30, Colab Tesla T4, anomalib 2.6.0, `preprocess=anomalib`, seeds 0/1/2, `test_public`
(150 images over **6** lighting conditions, **25** each). Commit `cbedb18`.

The table below is the runner's own output. The sections after it were added by hand.

Mean ± std over **3 seeds**, each aggregated separately (protocol §6):

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| overexposed | 0.8289 ± 0.0204 | 0.7781 ± 0.0017 | 0.4953 ± 0.0035 |
| regular | 0.8178 ± 0.0102 | 0.7819 ± 0.0017 | 0.4890 ± 0.0059 |
| shift_1 | 0.8622 ± 0.0154 | 0.7016 ± 0.0042 | 0.3670 ± 0.0023 |
| shift_2 | 0.8111 ± 0.0278 | 0.7772 ± 0.0028 | 0.4626 ± 0.0035 |
| shift_3 | 0.8200 ± 0.0306 | 0.7309 ± 0.0033 | 0.3896 ± 0.0027 |
| underexposed | 0.8022 ± 0.0139 | 0.7872 ± 0.0013 | 0.4836 ± 0.0060 |

## The result, stated plainly

**The best localisation in the grid, and the anchor is unambiguously usable here.** I-AUROC 0.802
to 0.862 across the six conditions, mean 0.824, with a spread of only 0.060 — the most stable
detection of any category run. `au_pro_030` averages 0.7595, **5.1× the measured random baseline
of 0.1487**, and `au_pro_005` averages 0.4479, **18× the 0.0249 baseline**. Vial still has the
highest single numbers (0.95 I-AUROC, 0.59 `au_pro_005` at `regular`), but Walnuts is the most
consistent.

## This kills resolution and aspect ratio as the explanatory variable

**Walnuts is 2448×2048. So is Wall Plugs, where the anchor is at or below chance in every
condition.** Same resolution, same 1.20:1 aspect ratio, same square `Resize([256, 256])`, opposite
outcomes — 0.824 mean I-AUROC here against 0.447 there. Whatever separates a working anchor from a
failing one on this dataset, it is not the geometry of the frame.

Six categories in, the full picture at this resolution and in general:

| category | W×H | aspect | anchor |
|---|---|---|---|
| Vial | 1400×1900 | 0.74:1 | works — 0.95 I-AUROC at `regular` |
| Wall Plugs | 2448×2048 | 1.20:1 | **fails** — ≤0.50 in all six |
| **Walnuts** | **2448×2048** | **1.20:1** | **works — 0.824 mean, best localisation** |
| Fruit Jelly | 2100×1520 | 1.38:1 | works — 0.79 `regular`, 0.85 best |
| Can | 2232×1024 | 2.18:1 | **fails** — 0.388 mean, below random at strict FPR |
| Sheet Metal | 4224×1056 | 4.00:1 | **fails** — at chance in all six |

The aspect-ratio mechanism registered on 2026-09-20 predicted Sheet Metal correctly and remains
the only candidate for that category. It explains nothing about the other two failures, and the
Wall Plugs / Walnuts pair shows it cannot be the general variable.

## The direction-shift pattern has its third reading, and its first from a working anchor

`au_pro_005`, intensity conditions against direction shifts:

| category | intensity (regular / over / under) | shifts | anchor |
|---|---|---|---|
| Wall Plugs | 0.1688 / 0.1723 / 0.1694 | 0.0985 / 0.0292 / 0.0753 | fails |
| Can | 0.0036 / 0.0028 / 0.0077 | 0.0001 / 0.0000 / 0.0000 | fails |
| **Walnuts** | **0.4890 / 0.4953 / 0.4836** | **0.3670 / 0.4626 / 0.3896** | **works** |

Walnuts means 0.4893 for the intensity conditions against 0.4064 for the shifts, and the same
ordering holds at the looser limit (0.7824 against 0.7366). **This is the first reading of the
pattern from a category where detection works**, so it no longer rests only on categories whose
anchor is broken.

**And detection is not affected at all here:** `shift_1` has the *best* I-AUROC of the six
(0.8622) while having the *worst* `au_pro_005` (0.3670). Changing the direction of the
illumination costs this category precise localisation and nothing else.

**It is a tendency and not a rule — two of five contradict it.** Vial's worst condition is
`underexposed` (0.49 `au_pro_005` against ~0.59 elsewhere), an intensity change, not a shift. Fruit
Jelly's `shift_1` is its *best* condition (0.3925 against 0.3820 at `regular`), though it has only
one shift to offer. Three of five categories in one direction, two against, is worth reporting as
a tendency with the exceptions named.

## Caveats

- **PatchCore is the full-shot anchor, not a zero-shot method.** Here it is a genuine ceiling, and
  a meaningful one to be measured against — unlike three of the six run so far.
- **One category is not a dataset result.** MVTec AD 2 has eight; this is the sixth, and the grid
  now stands at three usable anchors and three unusable.
- **Per-condition I-AUROC is coarse.** 60 normal and 90 anomalous images (Table 4) over six
  conditions gives **10 normal and 15 anomalous** each — 150 pairs, steps of 0.0067. AU-PRO is
  computed over pixels and is not subject to this, which is why its ± is an order of magnitude
  smaller.
- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio preservation.
  This is the pinned configuration the reproduction gate was passed at (protocol §7).

## Provenance

- shards: `results/mvtec_ad2/walnuts/shards`, and `MyDrive/mvtec_ad2_results/walnuts/`
- seeds: `0, 1, 2`
- coreset: **not captured**; it is **44236** by `floor(432 × 102.4)`, the relationship measured on
  2026-09-30. Third category in a row where the figure was not recorded, for the reason in
  `docs/next-steps.md`: it is provenance being read off a progress bar instead of stamped by the
  adapter.
- fit: **not captured**. The cost model predicts 5.1m per seed, ~15m for three.
- the summary needed `--summarise-only --max-bytes 12000000000`: 25 images × 2448×2048 is
  125,337,600 pooled pixels = **10.03 GB**, over the 6 GB pixel-metric guard
- commit: `cbedb18`
