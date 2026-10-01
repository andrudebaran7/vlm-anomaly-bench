# MVTec AD 2 — patchcore_ref on can

Run 2026-09-30, Colab Tesla T4, anomalib 2.6.0, `preprocess=anomalib`, seeds 0/1/2, `test_public`
(162 images over **6** lighting conditions, **27** each). Commit `d2d1cbd`.

The table below is the runner's own output. The sections after it were added by hand.

Mean ± std over **3 seeds**, each aggregated separately (protocol §6):

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| overexposed | 0.3685 ± 0.0085 | 0.3586 ± 0.0153 | 0.0028 ± 0.0029 |
| regular | 0.3259 ± 0.0160 | 0.3520 ± 0.0123 | 0.0036 ± 0.0016 |
| shift_1 | 0.3907 ± 0.0225 | 0.3047 ± 0.0118 | 0.0001 ± 0.0001 |
| shift_2 | 0.3815 ± 0.0210 | 0.2388 ± 0.0009 | 0.0000 ± 0.0000 |
| shift_3 | 0.4426 ± 0.0170 | 0.2394 ± 0.0090 | 0.0000 ± 0.0000 |
| underexposed | 0.4204 ± 0.0225 | 0.2970 ± 0.0104 | 0.0077 ± 0.0005 |

## The result, stated plainly

**This is the worst the anchor has done, and it is worse than uninformative.** Every one of the six
conditions is well below chance on I-AUROC — 0.326 to 0.443, mean 0.388, with the *training*
condition `regular` the worst of the six. And `au_pro_005` averages **0.0024 against a measured
random baseline of 0.0249**: about a tenth of what an anomaly map carrying no information at all
would score. Three of the six conditions are exactly 0.0000.

`au_pro_030` is the one metric above its baseline: mean 0.298 against 0.1487, roughly 2×. So at a
loose false-positive budget the map is doing something; at a strict one it is doing worse than
nothing.

**A number below its own random baseline is a claim about anti-correlation, not about difficulty**,
and it was not reported until two mechanical explanations had been ruled out.

## Two hypotheses tested and refuted, with the maps still on the machine

Both were run on 2026-09-30 in the session that produced the result, before the numbers were
written anywhere, and both outcomes are recorded here either way.

**1. Map/mask geometry — REFUTED.** Anomaly maps and ground-truth masks are both `(1024, 2232)`
for this category: same shape, same orientation, no transpose. Eight anomalous images at seed 0.

**2. The map lighting up the frame border — REFUTED.** The fraction of each map's top 1% of values
falling in the outer 5% band of the frame is **0.000 for all twelve images sampled**, six normal
and six anomalous. The band is ~19% of the area, so a diffuse top 1% would put ~0.19 there. The
map's hottest regions are entirely interior — on the can. That is the healthy behaviour: the
background around the object is the most consistent region across training images and therefore
the least anomalous.

**What the two together establish:** the map responds to the object and not to its defects, and on
five of eight sampled anomalous images the mean map value *inside* the ground-truth region is
**lower** than outside (ratios 0.796–0.880; the other three are 1.047–1.404). The anti-correlation
is in the scores, not in the geometry.

**What they do not establish:** why. No hypothesis is offered here. Testing one would mean
searching for a configuration that improves a number after seeing it, which §7 forbids; a
pre-registered mechanism, tested as a separate dated experiment, is the only route open, and none
was registered for this category.

## The pattern that is now on its second reading

**Direction shifts cost precise localisation more than intensity changes do.** On both `can` and
`wallplugs`, the three `shift_*` conditions score below the three intensity conditions on
`au_pro_005`:

| category | intensity (regular / over / under) | shifts (1 / 2 / 3) |
|---|---|---|
| Wall Plugs | 0.1688 / 0.1723 / 0.1694 | 0.0985 / 0.0292 / 0.0753 |
| Can | 0.0036 / 0.0028 / 0.0077 | 0.0001 / 0.0000 / 0.0000 |

The same ordering holds on `au_pro_030` here (intensity mean 0.336, shifts 0.261). Two categories
is not a finding, but it is the same direction twice, and both are categories where the anchor's
detection has failed — so the reading rests on localisation alone.

## Caveats

- **PatchCore is the full-shot anchor, not a zero-shot method.** On this category it is not a
  usable ceiling in any sense: a zero-shot method scoring 0.50 I-AUROC here **beats** it.
- **One category is not a dataset result.** MVTec AD 2 has eight; this is the fifth, and the third
  where the anchor is unusable.
- **Per-condition I-AUROC is coarse.** `test_public` gives this category 72 normal and 90
  anomalous images (Table 4) over six conditions, so each has **12 normal and 15 anomalous** — 180
  pairs, steps of 0.0056. Six conditions all well below 0.50, at three seeds each, is eighteen
  readings in the same direction; no significance test was pre-registered and none is claimed.
- **The diagnostics above are samples of eight and twelve images at seed 0**, chosen as the first
  rows of the shard, not sampled at random. They are enough to refute a shape defect and a border
  artifact; they are not measurements of the category.
- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio preservation,
  and Can is 2232×1024 (2.18:1). This is the pinned configuration the reproduction gate was passed
  at (protocol §7).

## Provenance

- shards: `results/mvtec_ad2/can/shards`, and `MyDrive/mvtec_ad2_results/can/`
- seeds: `0, 1, 2`
- coreset: **not captured**; it is **42188** by `floor(412 × 102.4)`, the relationship measured on
  2026-09-30. The fit output was not recorded, so this is the formula and not an observation —
  the third category in a row where a human was expected to copy a figure off a progress bar. See
  `docs/next-steps.md` for why the runner should stamp it instead.
- fit: **not captured**. The cost model predicts 4.6m per seed, ~14m for three.
- the summary needed no `--max-bytes`: 27 images × 2232×1024 is 61,710,336 pooled pixels =
  4.94 GB, under the 6 GB pixel-metric guard — the only one of the four remaining categories that
  fits, as projected
- commit: `d2d1cbd`
