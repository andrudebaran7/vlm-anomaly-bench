# Ground-truth-free threshold rule (spec)

**Date:** 2026-08-20
**Status:** approved, pre-implementation
**Produces:** one implementation plan — `docs/superpowers/plans/` — executable on CPU, no GPU session.
**Amends:** protocol §4 (v0.2.10 → v0.2.11).

`docs/next-steps.md` names this the one piece of unplanned work in the repo, and the only thing
standing between the study and M4 now that evaluation-server access exists. The official MVTec AD 2
metric is pixel-level SegF1 and the server takes **thresholded** anomaly maps, so a threshold must be
committed to without ever seeing ground truth. This repo computes `metrics.pixel_level.seg_f1max`,
which maximises F1 *over* thresholds by inspecting the ground truth — an oracle metric: legitimate and
comparable on `test_public`, strictly optimistic, and not what the server scores.

The paper claims the missing rule as contribution **C3** and claims private-split coverage as **C1**.
Both carry `\todo{withdraw if the threshold rule is not built before submission}` tripwires in
`vlm-anomaly-paper/sections/02-introduction.tex`. This spec is what removes them.

## The constraint that shapes everything

`validation` is **defect-free** and **regular lighting only** (41 images for Vial, verified on disk
2026-08-20; `docs/datasets-access.md` §layout). Two consequences that are not negotiable:

1. **No F1 is computable on `validation`.** There are no positives. The only quantity estimable there
   is the distribution of scores on normal pixels. Every rule below is therefore a statement about a
   target false-positive rate, because that is the only thing the calibration data can speak to.
2. **The calibration split never sees a lighting shift, and half the private split is nothing but
   lighting shift** (`test_private_mixed`). Whether a threshold survives that transfer is not an
   implementation detail — it is the finding C3 reports.

Maps are on each method's own scale (protocol v0.2.6, no per-image rescaling of scored maps), so a
threshold is necessarily per **(method, category)**. Nothing here is shared across methods.

## 1. The framework: one primitive, two axes

The three candidate rules are not three algorithms. They are three instantiations of one primitive:
*transform the scores, pool them, cut at a quantile*.

```
quantile_threshold(transform, source, alpha)
```

- **`transform`** — `identity` (the method's raw score) or `robust_z` (per image,
  `(s - median(s)) / MAD(s)`).
- **`source`** — `validation` (defect-free by dataset construction, which is metadata, not ground
  truth) or `test_split` (the test images themselves, **inputs only, never labels**).
- **`alpha`** — the target false-positive rate on normal pixels. The only free parameter, because it
  is the only quantity the calibration data can estimate.

| Rule | `transform` | `source` | Fitted parameter |
|---|---|---|---|
| **A · `global_quantile`** | `identity` | `validation` | one scalar threshold per (method, category) |
| **B · `per_image_robust_z`** | `robust_z` | `validation` | one scalar `k` per (method, category) |
| **C · `transductive_quantile`** | `identity` | `test_split` | none; resolved at apply time |

Rule B is calibrated exactly as A is, in standardised space: pool `z` across every validation image,
take the `(1 - alpha)` quantile, and that value **is** `k`. At apply time the threshold for a test
image is `median(s) + k * MAD(s)` computed on that image alone. Same code path, different transform.

### How each one fails, stated in advance

- **A** assumes the score scale is stable between `validation` and test. It breaks precisely where
  `test_private_mixed` differs from `validation`.
- **B** absorbs any per-image shift in level and spread, at the cost of assuming the defective region
  does not move that image's own median and MAD. The MAD's breakdown point is 50%, so this holds for
  any realistic defect size.
- **C** adapts to a shift affecting the split as a whole, but pools normal and anomalous pixels
  together, so its realised FPR deviates from `alpha` by roughly the anomalous-pixel prevalence.

Reporting all three against the oracle is what makes C3 a result on an open problem rather than an
assertion that one rule is reasonable. The challenge organisers name threshold selection "a challenge
often not yet considered within the scientific community but indispensable for deployment"
(arXiv:2509.17615, recorded in protocol §4 v0.2.10).

### Two clarifications that prevent a misreading

- **B does not violate v0.2.6.** The continuous map submitted to the server stays on the method's own
  scale, unrescaled. Only the *binarisation* is per-image. This looks superficially like the per-image
  `[0,1]` normalisation §4 forbids, and it is not: v0.2.6 protects the cross-image ranking that
  P-AUROC, AU-PRO and SegF1max depend on, and a per-image threshold changes no score. The protocol
  amendment says this explicitly.
- **C is transductive.** It uses the test *inputs*, never the labels, which is ground-truth-free by
  this protocol's definition and legitimate for a batch-submission benchmark. It is nonetheless a
  different access assumption from A and B, so it is labelled transductive in every table it appears
  in.

### Alpha

**`alpha = 1e-3`, pre-registered, one value.** Justified as a stated deployment operating point and
nothing more: *no more than one pixel in a thousand flagged on a defect-free part.*

It is deliberately **not** derived from the defect-area fraction of the data. That statistic exists
only in `test_public`'s ground-truth masks, and using it to choose `alpha` would be calibrating on
test. If a future version wants an empirical justification, the number must come from a primary source
recorded in `vlm-anomaly-paper/docs/verified-literature-facts.md` **before** it is used here — the
standing rule of this project (`MEMORY.md`, "record provenance before writing").

**Sensitivity to `alpha` is reported** as a curve over the fixed decade grid
`{1e-4, 1e-3, 1e-2, 1e-1}` on `test_public`, pre-registered here so the grid itself cannot be chosen
after seeing the curve. Reporting a curve is not selecting on it, so long as the submitted `alpha` was
fixed first — which this spec does, and the protocol amendment records with a date.

## 2. The designated rule: B

**`per_image_robust_z` is designated for the private-split submission**, fixed here before any result
exists. §7 of the protocol allows one submission per method, so this choice cannot be revisited after
seeing public-split numbers.

The justification is a priori and does not depend on any measurement:

1. What distinguishes the private split is `test_private_mixed`'s lighting variation, and B is the
   only candidate whose calibration is invariant to a per-image shift in level and spread.
2. B is **inductive** — it needs one image at a time, not the whole split — which is the deployment
   setting §1 of the paper argues zero-shot methods exist to serve. C would score better by adapting
   to the split it is scoring, and that is exactly why it is not designated.

A and C are still computed, reported and compared on `test_public`. Only B is submitted.

## 3. `seg_f1_at`, and why the memory conflict dissolves

```python
seg_f1_at(masks, amaps, threshold) -> float     # threshold: scalar, or one per image
normal_pixel_fpr_at(masks, amaps, threshold) -> float
```

`seg_f1_at` accumulates TP/FP/FN one image at a time and computes F1 once at the end. Predicted
positive is `amap >= threshold` — `>=`, fixed and documented, because it decides the degenerate cases.

This pools pixels **over the complete category**, which is literally the official definition recorded
in protocol §4 v0.2.10: precision and recall over the complete set of pixels in the test set, not
averaged over individual images. At a fixed threshold nothing needs to be sorted, only counted, so the
working set is one mask plus one map at a time — under 20 MB at Vial's 1400x1900 resolution, against
the 29.8 GB that `pixel_metrics` must refuse for the same 140 images.

So the apparent conflict between the official definition and the 6 GB budget is not worked around: it
does not exist. `seg_f1max` is expensive and per-lighting-condition because it sorts; the official
metric turns out to be the cheap one.

**Degenerate cases, all fixed:**

- No ground-truth positives in the group → `0.0`, mirroring `seg_f1max`.
- Nothing predicted (`TP + FP == 0`) → `0.0`; precision is undefined and F1 with it.

**The invariant that ties the two metrics together**, tested as a property:
`seg_f1_at(m, a, t) <= seg_f1max(m, a)` for every `t`.

`normal_pixel_fpr_at` reports the **realised** FPR on test against the `alpha` **targeted** on
validation. For A and B under lighting shift, that gap is itself a headline number of C3. It needs
masks to know which pixels are normal, so it is computable on `test_public` only — never on the
private splits, where it is exactly the kind of feedback §7 forbids iterating against.

### Calibration is streaming too

The `(1 - alpha)` quantile of a pooled set is the k-th largest value with `k = ceil(alpha * N)`. For
Vial's validation split at `alpha = 1e-3` that is about 109,000 values out of 109 Mpx — 0.4 MB of
running state against 436 MB to pool the split. Calibration is therefore two streaming passes:

1. Read shapes only (`np.load(..., mmap_mode="r").shape`, the pattern `pixel_metrics` already uses) to
   get `N`, hence `k`.
2. Stream the maps, applying `transform` per image, maintaining a running top-`k`.

The result is the **exact** quantile in `O(k)` memory, not an approximation, and the same primitive
serves rule C at apply time over the test split. No memory guard is needed on either path.

## 4. The calibration artifact

One YAML per (dataset, method) under `configs/thresholds/`, **committed to git**:

```yaml
dataset: mvtec_ad2
method: intensity_baseline
protocol_version: "0.2.11"
alpha: 0.001
calibrated_on:
  split: validation
  n_images: 41
  lighting: [regular]
run_id: <run_id of the validation run>
calibrated_at: 2026-08-20
categories:
  vial:
    global_quantile:      {threshold: 12.34}
    per_image_robust_z:   {k: 8.7}
    transductive_quantile: {}        # no fitted parameter
designated_for_submission: per_image_robust_z
```

Committing it is what makes "pre-registered" auditable: a file in git with a date, not a promise. A
submission whose artifact was not committed *before* the submission date is a protocol violation that
can be checked mechanically against the log.

`artifact.py` refuses to load an artifact whose `protocol_version` does not match the protocol in
force, or whose `designated_for_submission` names a rule absent from `categories`.

## 5. Module layout and integration

**New package `src/vlmab/threshold/`:**

- `rules.py` — the transforms (`identity`, `robust_z`), the streaming top-`k` quantile primitive,
  `quantile_threshold`, and the three named rules. Pure numpy, no I/O.
- `artifact.py` — read, write and validate the calibration YAML.

**Metrics** go in `src/vlmab/metrics/pixel_level.py` beside the others: `seg_f1_at`,
`normal_pixel_fpr_at`.

**Aggregation.** A separate path in `aggregate.py`, deliberately *not* inside `pixel_metrics`:

```python
threshold_metrics(df, calibration) -> dict     # streaming; no max_bytes; whole category
```

`pixel_metrics` is untouched: it still sorts, so it stays per-lighting-condition and keeps its guard.
The two paths are separated by their real cost, which is also why the official metric can pool a whole
category and the oracle cannot.

Both granularities are reported and labelled: `seg_f1_at__{A,B,C}` and `fpr_at__{A,B,C}` over the
**complete category** (the official definition) and **per lighting condition** (the shift analysis).
`seg_f1max` remains per-condition and marked **oracle**; §4's existing rule that the two never share
an unmarked column stands.

**New CLI** `scripts/calibrate_threshold.py`. End-to-end flow:

```
run_eval.py --split validation --maps-dir ...      # already supported, unchanged
calibrate_threshold.py --run ... --method M --alpha 1e-3
    -> configs/thresholds/mvtec_ad2__M.yaml        # commit this: it is the pre-registration
run_eval.py --split test_public --maps-dir ...
aggregate.threshold_metrics(df, load_artifact(...))    # library call, as aggregate already is
```

`aggregate` has no CLI today (`scripts/make_tables.py` is an M5 stub), and this spec does not add one.
Aggregation is called from the Colab notebook exactly as it is now; only `calibrate_threshold.py` is a
new entry point, because it is the step whose output gets committed.

## 6. Protocol amendment (v0.2.11)

§4 is rewritten to pre-register, with a dated Changelog entry pointing at this spec:

- the two-axis framework and the three named rules;
- `alpha = 1e-3`, its deployment justification, and the explicit prohibition on deriving it from test
  masks;
- **B as the designated submission rule**, with the a priori justification of §2 above;
- the oracle/rule separation in tables, extending the existing `seg_f1max` marking rule;
- the clarification that a per-image *threshold* is not the per-image *normalisation* v0.2.6 forbids;
- that C is transductive and labelled as such.

## 7. Verification

TDD, in the repo's existing style — a naive reference plus property tests.

- `seg_f1_at` against a naive whole-pooling implementation on small arrays.
- The invariant `seg_f1_at(t) <= seg_f1max` over random thresholds.
- The streaming top-`k` quantile against `np.quantile` on data small enough to pool, including the
  `k = 1` and `k >= N` edges.
- Calibration recovers the target: on synthetic maps of known distribution, the realised FPR on
  held-out normals is `alpha` within tolerance.
- **The test that encodes the scientific claim:** applying an affine transform to every pixel of an
  image leaves **B**'s binary mask identical and **changes A**'s. If that invariance ever breaks, B
  has lost its reason to exist and its designation must be revisited.
- `MAD == 0` (a constant image): threshold `+inf`, nothing predicted, and the image counted in a
  `degenerate_mad` tally the apply path returns alongside the metrics. Fixed and reported rather than
  left to divide by zero or silently dropped.
- Artifact validation: version mismatch and a designated rule missing from `categories` both raise.

**End to end on real data:** Vial with `intensity_baseline`, CPU only — the method is pure numpy and
Vial is the one category on disk, so the whole chain runs today without a GPU session. This is a
**documented manual step, not CI**: the tests build synthetic trees (`tests/mvtec_tree.py`) and Vial's
0.77 GB is not in git. CI runs the same chain over the synthetic tree.

## 8. Out of scope

- **Submission packaging.** Writing the server's payload (thresholded plus continuous maps in its
  expected format) is a separate piece of work. The exact format is unverified until first login
  (`docs/datasets-access.md` leaves it as a checkbox), and writing it now would mean guessing an
  upstream API, which protocol §3 forbids.
- **Morphological post-processing** of the binary mask (opening, minimum connected-component area).
  It would raise SegF1 by removing isolated-pixel noise, but it is orthogonal to how the threshold is
  chosen and adds hyperparameters that would themselves need calibrating. If it is added later it is
  its own amendment.
- **Removing the `\todo{withdraw}` tripwires** from C1 and C3. That is a commit in the paper repo,
  after this lands and after a first calibration actually runs.

## 9. What this unblocks

M4 stops being blocked on unbuilt work. It remains blocked on the out-of-scope submission packaging
and on each method's VisA gate, which is the pre-existing ordering in `docs/next-steps.md`. The value
delivered here is independent of any Colab session: after this lands, the study can report SegF1 at a
pre-registered rule alongside the oracle, on any category it has maps for.
