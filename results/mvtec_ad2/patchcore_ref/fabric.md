# MVTec AD 2 — patchcore_ref on fabric

Run 2026-10-01, Colab Tesla T4, anomalib 2.6.0, `preprocess=anomalib`, seeds 0/1/2, `test_public`
(156 images over **6** lighting conditions, **26** each). Commit `0fd705a`.

**The eighth and last category of M3.**

The table below is the runner's own output. The sections after it were added by hand.

Mean ± std over **3 seeds**, each aggregated separately (protocol §6):

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| overexposed | 0.5131 ± 0.0185 | 0.1629 ± 0.0061 | 0.0095 ± 0.0025 |
| regular | 0.5152 ± 0.0160 | 0.1580 ± 0.0104 | 0.0083 ± 0.0026 |
| shift_1 | 0.5414 ± 0.0273 | 0.1800 ± 0.0055 | 0.0172 ± 0.0061 |
| shift_2 | 0.6848 ± 0.0105 | 0.1170 ± 0.0108 | 0.0060 ± 0.0013 |
| shift_3 | 0.5636 ± 0.0242 | 0.1071 ± 0.0144 | 0.0020 ± 0.0009 |
| underexposed | 0.5556 ± 0.0299 | 0.1509 ± 0.0137 | 0.0024 ± 0.0015 |

## The result, stated plainly

**Localisation is at the random baseline, and detection is weakly above chance.** `au_pro_030`
averages **0.1460 against the measured random baseline of 0.1487 — 0.98×**, which is to say the
anomaly maps are indistinguishable from maps carrying no information. `au_pro_005` averages 0.0076,
**0.30×** its 0.0249 baseline, below uninformative. Meanwhile all six conditions sit above 0.50
I-AUROC, mean 0.562, with `shift_2` at 0.6848.

**This is a failure mode the other seven categories did not produce.** Wall Plugs and Rice fail
detection while localising above random. Fabric does the opposite: it has a weak but consistent
image-level signal with no usable localisation at all. Can is the only other category below its
random baseline, and only at the strict limit.

**`shift_2` pulls the two metrics apart inside this category too**: the best I-AUROC of the six
(0.6848) and the second-worst `au_pro_030` (0.1170). The same shape as Fruit Jelly's and Walnuts'
`shift_1`.

## What this does to the eight-category picture

Per-category means, pixel metrics as multiples of their measured random baselines:

| category | i_auroc | au_pro_030 | ×rnd | au_pro_005 | ×rnd | anchor |
|---|---|---|---|---|---|---|
| Vial | 0.910 | 0.8414 | 5.66 | 0.5714 | 22.95 | usable |
| Walnuts | 0.824 | 0.7595 | 5.11 | 0.4479 | 17.99 | usable |
| Fruit Jelly | 0.783 | 0.6394 | 4.30 | 0.3865 | 15.52 | usable |
| **Fabric** | **0.562** | **0.1460** | **0.98** | **0.0076** | **0.30** | **no** |
| Rice | 0.519 | 0.3536 | 2.38 | 0.1695 | 6.81 | no |
| Sheet Metal | 0.510 | 0.2523 | 1.70 | 0.0677 | 2.72 | no |
| Wall Plugs | 0.447 | 0.4194 | 2.82 | 0.1189 | 4.78 | no |
| Can | 0.388 | 0.2984 | 2.01 | 0.0024 | 0.10 | no |

**Five of the eight categories have no usable full-shot anchor** under the pinned configuration.
Three do: Vial, Walnuts, Fruit Jelly.

### It also corrects a claim made one day earlier, on seven categories

Rice's report states that ordering the categories by I-AUROC and by `au_pro_005` gives the same
sequence apart from one adjacent transposition. **Fabric breaks that.** It is **4th by detection
and 7th by localisation** — a three-place shift — and the two orderings now differ by a Spearman
rank correlation of **0.833** over eight categories rather than being near-identical:

* by I-AUROC: Vial > Walnuts > Fruit Jelly > **Fabric** > Rice > Sheet Metal > Wall Plugs > Can
* by `au_pro_005`: Vial > Walnuts > Fruit Jelly > Rice > Wall Plugs > Sheet Metal > **Fabric** > Can

The agreement is still strong and the top three are identical in both. What is no longer true is
that the two metrics rank the categories *the same way*: a category can carry an image-level signal
with no localisation (Fabric) or localise with no image-level signal (Wall Plugs, Rice). **Both
directions occur, which is a stronger argument for reporting the two metrics separately than
either direction alone.**

## Caveats

- **PatchCore is the full-shot anchor, not a zero-shot method.** Its localisation here is at
  random, so there is no localisation ceiling to measure anything against on this category.
- **Eleven normal images per lighting condition.** `test_public` gives fabric 66 normal and 90
  anomalous (Table 4) over six conditions — 165 pairs, steps of 0.0061. The detection numbers are
  consistently above 0.50 but modest; no significance test was pre-registered and none is claimed.
- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio preservation.
  Fabric is 2448×2048 (1.20:1) — the same frame as Walnuts (the best anchor in the grid), Wall
  Plugs and Rice (both failures). **At this one resolution the anchor works on one category of four**,
  which closes the geometry question for good.
- **One category is not a dataset result, but eight are.** This completes the public-split grid for
  the anchor.

## Provenance

- shards: `results/mvtec_ad2/fabric/shards`, and `MyDrive/mvtec_ad2_results/fabric/`
- seeds: `0, 1, 2`
- coreset: **not captured**; it is **39628** by `floor(387 × 102.4)`, the relationship measured on
  2026-09-30. Fifth category running where the figure was not recorded, which is now logged as a
  defect in `docs/next-steps.md` rather than an oversight.
- fit: **not captured**. The cost model predicts 4.1m per seed, ~12m for three.
- the summary needed `--summarise-only --max-bytes 12000000000`: 26 images × 2448×2048 is
  130,351,104 pooled pixels = **10.43 GB**, over the 6 GB pixel-metric guard
- upload: 10.84 GB to Drive by hand. The in-session `wget` route verified on 2026-10-01
  (`docs/datasets-access.md`) was not used for this run.
- commit: `0fd705a`
