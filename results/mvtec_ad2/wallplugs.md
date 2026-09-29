# MVTec AD 2 — patchcore_ref on wallplugs

Run 2026-09-29, Colab Tesla T4, anomalib 2.6.0, `preprocess=anomalib`, seeds 0/1/2, `test_public`
(150 images over **6** lighting conditions, **25** each). Commit `2db64c2`.

The table below is the runner's own output. The sections after it were added by hand.

Mean ± std over **3 seeds**, each aggregated separately (protocol §6):

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| overexposed | 0.4378 ± 0.0192 | 0.4328 ± 0.0124 | 0.1723 ± 0.0082 |
| regular | 0.4600 ± 0.0133 | 0.4253 ± 0.0121 | 0.1688 ± 0.0062 |
| shift_1 | 0.4556 ± 0.0168 | 0.5132 ± 0.0251 | 0.0985 ± 0.0246 |
| shift_2 | 0.3956 ± 0.0559 | 0.2773 ± 0.0088 | 0.0292 ± 0.0055 |
| shift_3 | 0.4356 ± 0.0434 | 0.4419 ± 0.0214 | 0.0753 ± 0.0132 |
| underexposed | 0.5000 ± 0.0067 | 0.4260 ± 0.0097 | 0.1694 ± 0.0054 |

## The result, stated plainly

**Detection fails, and localisation does not.** Every one of the six conditions is at or below
0.50 I-AUROC — mean 0.447, best 0.5000 (`underexposed`, exactly 75/150 pairs), worst 0.3956
(`shift_2`). Meanwhile AU-PRO@30% runs 0.2773–0.5132 against a **measured random baseline of
0.1487**, so the anomaly maps are locating the defects. The image-level score, which is what
I-AUROC reads, is not ranking the images that contain them.

This is the sharpest instance of the dissociation Fruit Jelly showed in one condition, and here
it is the whole category.

**The random baseline is measured, not asserted.** `tests/test_au_pro_random_baseline.py` runs
uninformative maps through this repo's own `au_pro` and gets 0.1487 at the 30% limit and 0.0249 at
5%, matching the `L/2` the metric's definition implies. Without that reference an AU-PRO number
cannot be read at all, which is why it was added the same day as this result.

**`shift_2` is the worst condition on all three metrics, and its precise localisation is gone.**
Its `au_pro_005` is 0.0292 against the random baseline's 0.0249 — within 17% of uninformative.
At the looser 30% limit it still scores 0.2773, so something survives; at a strict false-positive
budget, nothing usable does.

**The three intensity conditions and the three direction shifts separate cleanly at strict FPR.**
`au_pro_005` for `regular`/`overexposed`/`underexposed` is 0.1688/0.1723/0.1694 — three values
inside 0.0035 of each other. For `shift_1`/`shift_2`/`shift_3` it is 0.0985/0.0292/0.0753, a mean
2.5× worse. Changing the *intensity* of the illumination costs this category almost nothing;
changing its *direction* costs it precise localisation. At the 30% limit the shifts are erratic
rather than uniformly worse (`shift_1` is the best of all six, `shift_2` the worst), so the effect
is specifically about how tight a false-positive budget the map can hold.

## What this rules out

**Aspect ratio does not explain anchor failure.** Wall Plugs is 2448×2048 — 1.20:1, the mildest
reshaping in the dataset apart from Vial's, where the anchor works well. The pre-registered
mechanism for Sheet Metal's failure (a 4:1 frame compressed to square) has nothing to say here.
So the study now has **two categories of eight where the full-shot anchor is unusable, and only
one of them has an extreme aspect ratio.** Whatever is happening on Wall Plugs is a second,
separate failure, and Sheet Metal's mechanism remains untested as a cause of its own.

**It is not a lighting-robustness story either.** The failure is present in `regular`, the
condition the training images were captured under, at 0.4600. Lighting shift makes a bad number
slightly worse; it did not cause it.

## Caveats

- **PatchCore is the full-shot anchor, not a zero-shot method.** On this category it is not a
  usable ceiling: a zero-shot method scoring 0.50 here is not "matching the anchor".
- **One category is not a dataset result.** MVTec AD 2 has eight; this is the fourth.
- **Below-chance is consistent here but not tested.** `test_public` gives this category 60 normal
  and 90 anomalous images (Table 4) over six conditions, so each condition has **10 normal and 15
  anomalous** — 150 pairs, steps of 0.0067. Six conditions all at or below 0.50, at three seeds
  each, is a consistent pattern; with ten normal images per condition it is not a significance
  test, and none was pre-registered. Stated as what it is: eighteen readings, none above chance.
- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio preservation.
  This is the pinned configuration the reproduction gate was passed at (protocol §7).

## Provenance

- shards: `results/mvtec_ad2/wallplugs/shards`, and `MyDrive/mvtec_ad2_results/wallplugs/`
- seeds: `0, 1, 2`
- coreset: **not captured** for this category. Predicted 30002 by the pattern that held three
  times (`floor(293 × 102.4) − 1 = 30002`); the fit's progress bar was not recorded, so nothing
  here confirms or refutes it. The open question of whether that figure is the memory bank or an
  off-by-one in the bar is therefore still open — see `docs/next-steps.md`.
- the summary needed `--summarise-only --max-bytes 11000000000`: 25 images × 2448×2048 is
  125,337,600 pooled pixels = **10.03 GB**, over the 6 GB pixel-metric guard. The docs had
  projected 8.02 GB for this resolution from ~20 images per condition; this category has 25, and
  the projection was 25% low. Corrected in `docs/datasets-access.md`.
- commit: `2db64c2`
