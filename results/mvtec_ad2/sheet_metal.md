# MVTec AD 2 — patchcore_ref on sheet_metal

Run 2026-09-21, Colab Tesla T4, anomalib 2.6.0, `preprocess=anomalib`, seeds 0/1/2, `test_public`
(114 images over **6** lighting conditions, 19 each). Commit `a7bc7bd`.

The table below is the runner's own output. The sections after it were added by hand.

Mean ± std over **3 seeds**, each aggregated separately (protocol §6):

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| regular | 0.5500 ± 0.0000 | 0.2126 ± 0.0045 | 0.0570 ± 0.0015 |
| overexposed | 0.5111 ± 0.0192 | 0.2136 ± 0.0048 | 0.0574 ± 0.0016 |
| underexposed | 0.4889 ± 0.0347 | 0.2160 ± 0.0049 | 0.0569 ± 0.0008 |
| shift_1 | 0.4611 ± 0.0096 | 0.2938 ± 0.0046 | 0.0800 ± 0.0016 |
| shift_2 | 0.5611 ± 0.0347 | 0.2756 ± 0.0032 | 0.0737 ± 0.0024 |
| shift_3 | 0.4889 ± 0.0255 | 0.3021 ± 0.0048 | 0.0814 ± 0.0002 |

## The result, stated plainly

**PatchCore is at chance on this category.** I-AUROC ranges 0.46–0.56 where random is 0.50, and
that includes `regular`, the condition the training images were captured under. The full-shot
anchor — the ceiling every zero-shot number in this study is measured against — does not detect
anomalies on Sheet Metal at all.

**This is not the lighting-shift story.** On Vial, `regular` 0.95 degraded to `underexposed` 0.87:
a real effect with a clear direction. Here every condition is indistinguishable from every other
and all of them are indistinguishable from chance. Whatever is happening precedes the lighting.

**The ± on `regular` is granularity, not stability.** `test_public` gives this category 24 normal
and 90 anomalous images (Table 4), so each condition has **4 normal and 15 anomalous** — 60
pairs, and I-AUROC can only take values k/60, in steps of 0.0167. `regular`'s 0.5500 is exactly
33/60 at all three seeds. A ±0.0000 here means the metric could not resolve a difference, not
that the method was stable.

## The pre-registered explanation, and why it is not a post-hoc one

**Sheet Metal is 4224×1056 — a 4:1 frame — and anomalib 2.6.0's pre-processor is a fixed square
`Resize([256, 256])` with no aspect-ratio preservation.** That compresses this category
**16.5× horizontally** against 4.1× vertically. A defect 50 px across in the original becomes
**3 px wide**, and PatchCore's 32×32 feature grid puts one patch over every 8×8 input pixels — so
such a defect does not fill half of a single patch in the direction it is measured.

**This mechanism was written down on 2026-09-19 and recorded again on 2026-09-20, before this run
existed**, in `docs/next-steps.md`, `docs/datasets-access.md` and the paper repo's provenance —
and Sheet Metal was named there as the case where it would be most extreme, because it is the
only 4:1 category. That ordering is what makes it a hypothesis rather than an excuse.

**It is still not confirmed.** Nothing here rules out the defects simply being beyond a
256×256 representation regardless of aspect ratio, or something specific to this category's
surface. The result is reported as it stands.

## What is NOT being done about it

**Nothing is reconfigured.** Protocol §7 forbids changing a configuration in response to a
result, and the square resize is the pinned configuration PatchCore's reproduction gate was
passed at. Changing it now would invalidate the comparison to every number already recorded.

What §7 does allow is testing the pre-registered hypothesis as a separate, dated experiment whose
outcome is recorded either way — the same footing the CenterCrop hypothesis had on MVTec AD
classic, which was tested and **refuted**. That has not been done here and is not scheduled.

## What it means for the study

The anchor defines the ceiling. On a category where the anchor sits at chance, a zero-shot
method's number is being compared against a ceiling that measures nothing, and "close to
PatchCore" stops meaning "close to the achievable". **Sheet Metal needs this stated wherever its
zero-shot results appear**, and it may argue for reporting the category with the anchor's failure
attached rather than as an ordinary row.

## Caveats

- **PatchCore is the full-shot anchor, not a zero-shot method.**
- **One category is not a dataset result.** MVTec AD 2 has eight.
- **Four normal images per lighting condition** makes per-condition I-AUROC coarse; see above.
- Compared against nothing in the dataset paper: its "below 60% average AU-PRO" is a
  dataset-level average under its own protocol, not a per-category figure.

## Provenance

- shards: `results/mvtec_ad2/sheet_metal/shards`, and `MyDrive/mvtec_ad2_results/sheet_metal/`
- seeds: `0, 1, 2`; coreset 14027 indices each — `floor(137 × 102.4)`, the 32×32 grid
- fit: 29s / 29s / 30s, against the 31s the cost model predicted
- the summary needed `--summarise-only --max-bytes 8000000000`: 19 images × 4224×1056 is
  84,750,336 pooled pixels = 6.78 GB, over the 6 GB pixel-metric guard
- commit: `a7bc7bd`
