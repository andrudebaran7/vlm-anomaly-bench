# Evaluation protocol (frozen before results)

The point of freezing this document is credibility: every choice below is made **before** seeing a
single result, so no metric or threshold can be quietly tuned to make one method look good.
Any later change gets a dated entry in the Changelog at the bottom.

## 1. Task definition

Image-level anomaly detection + pixel-level anomaly localisation, **zero-shot** (no target-category
training images) unless a method's protocol explicitly defines an auxiliary-training regime
(AnomalyCLIP, AdaCLIP: trained on auxiliary datasets that must NOT overlap with any test dataset —
overlap table required in the paper).

## 2. Datasets & splits

- **MVTec AD 2** — 8 scenarios, >8,000 high-resolution images. Defect-free train and validation
  sets; the test set is split into a public part with pixel-precise ground truth and a private part
  whose ground truth is withheld. Local metrics are computed on the public part only. The private
  part is scored solely through the official evaluation server (benchmark.mvtec.com).
  Licensed CC BY-NC-SA 4.0. Split names verified 2026-07-23 from the official download page:
  `train`, `validation`, `test_public`, `test_private`, `test_private_mixed`.

  `test_public` carries images from **every lighting condition** with pixel ground truth, so the
  lighting-robustness analysis is computed locally. `test_private` (regular lighting) and
  `test_private_mixed` (the same scenes under varied lighting) are a controlled paired
  comparison whose ground truth is withheld; they are reported only if server access is
  obtained. Their absence narrows the lighting result, it does not remove it.

  Samples from `test_private` and `test_private_mixed` carry no label, since their ground truth
  is withheld. The loader marks them `label = -1` and every accuracy metric refuses to run on
  them rather than silently treating unknown as normal.
- **Real-IAD Variety** — out of scope for this study. The dataset is publicly available (no
  application required), but its scale (198,950 images; the parent Real-IAD release is 622 GB, 53 GB
  for the 1024px variant) is incompatible with the Colab compute budget. Recorded as a stated
  limitation, not an omission.
- **VisA + MVTec AD (classic)** — used only to validate our re-implementations against published
  numbers (tolerance ±1.0 I-AUROC). If we can't reproduce a method's published VisA/MVTec numbers,
  its frontier results are flagged as such in every table.

## 3. Methods & implementations

Priority order: (1) official code, pinned commit; (2) anomalib implementation, pinned version;
(3) re-implementation (last resort, flagged). Prompts for CLIP-based methods are taken verbatim
from the original papers. The governing principle: what is forbidden is prompt content **we** write,
reword or tune per category. Per-category content taken **verbatim from the method's own published
source** — WinCLIP's object noun from its paper's prompt ensemble, SAA+'s per-object prompts from its
repo — is not "engineering"; it is required for faithful reproduction, and omitting it would measure
a degraded method the source never reports. The MLLM baseline's own commitment is stricter than this
floor, and stays binding regardless: one fixed structured prompt + one fixed scoring rubric for **all**
datasets and categories, published in `configs/methods/mllm_qwen.yaml`, with no per-category variation
of any kind — the MLLM has no published per-category prompt to reproduce, so there is nothing verbatim
to substitute.

- **Colour channels.** MVTec AD 2 images are grayscale in at least one category (Vial: 8-bit,
  1400x1900). Every method in this study expects 3-channel RGB input, so the loader converts
  with PIL's `convert("RGB")`, which replicates the single channel and passes an already-RGB
  image through unchanged. This is applied identically to every method and every category, so
  it cannot advantage one method over another. It is recorded here because it is a
  preprocessing decision that touches every reported number.

- **Anomaly-map storage precision.** The runner persists each anomaly map to disk as
  **float16** (`runner.run_evaluation`); aggregation reads it back and upcasts to float32
  before any metric sees it. This halves the map footprint, which is what makes a category of
  2.66 MP maps fit a Colab session's disk and memory at all. Measured effect on the reported
  numbers, comparing every pixel metric computed from float32 maps against the same maps
  round-tripped through float16 (5 random fixtures, 6 images of 256x256, multiple regions per
  image): P-AUROC 1.4e-06, AU-PRO@0.3 3.9e-05, AU-PRO@0.05 2.4e-05, SegF1max 8.0e-04 — worst
  case an order of magnitude inside the ±1.0-point (0.01) tolerance this protocol uses
  elsewhere, and applied identically to every method and category. Recorded here, like the
  colour-channel rule above, because it is a decision that touches every pixel-level number.

- **Full-shot anchors.** A full-shot method (the PatchCore anchor) builds a per-category memory
  bank from that category's defect-free `train` split before scoring its test images. It declares
  `zero_shot = False`; the runner calls `fit(train_images, category)` once per category. This is the
  standard PatchCore regime and is fixed here so the anchor's numbers are comparable across
  categories.

- **AnomalyCLIP (auxiliary-trained).** AnomalyCLIP learns object-agnostic prompts on an auxiliary AD
  dataset and scores zero-shot; the auxiliary data must not overlap the test set (§3.1 credibility
  requirement). Provenance is the official repo at a pinned commit (anomalib does not ship it). The
  checkpoint is chosen per test set so the auxiliary data is clean: the VisA-trained checkpoint for
  MVTec AD 2 (conservative, avoiding MVTec-family domain proximity), the MVTec-AD-trained checkpoint
  for the VisA reproduction. Full audit: docs/anomalyclip-overlap-audit.md.

- **AdaCLIP (auxiliary-trained).** AdaCLIP adapts CLIP with hybrid prompts — a static component learned
  on auxiliary AD data and fixed in the checkpoint, plus a per-image dynamic component — and scores
  zero-shot; the auxiliary data must not overlap the test set (§3.1 credibility requirement).
  Provenance is the official repo at a pinned commit (anomalib does not ship it). The checkpoint is
  chosen per test set on the same rule as AnomalyCLIP, so the auxiliary data is clean: the VisA-trained
  checkpoint for MVTec AD 2 (conservative, avoiding MVTec-family domain proximity), the
  MVTec-AD-trained checkpoint for the VisA reproduction. If the repo publishes only one checkpoint, the
  pre-registered contingency in docs/adaclip-overlap-audit.md applies: the available checkpoint is used
  and the resulting domain proximity is recorded as a caveat on every table, never reported as clean.
  Full audit: docs/adaclip-overlap-audit.md.

- **SAA+ (training-free; per-category prompts sourced from the repo).** SAA+ is a cascade of two frozen
  foundation models (GroundingDINO region proposals refined by SAM). Nothing is trained by the method,
  so no auxiliary-training overlap audit applies — SAA+ appears in the §3.1 table as *training-free, no
  auxiliary data*, so that table has no silent gaps. **This bullet applies §3's verbatim-sourcing
  principle to SAA+, whose per-object prompts are published in the official repo rather than in a
  paper.** SAA+'s contribution is hybrid prompt regularization: per-object defect language expressions
  and object-specific property constraints. Running it with a single generic prompt would measure a
  method its own paper does not report. Applying the principle here is conditional: the per-object
  prompts are taken **verbatim from the official repo at the pinned commit**, recorded with their exact
  file provenance in configs/methods/saa_prompts.yaml, committed before the first scoring run, and never
  adjusted after seeing a result. Where the repo publishes no prompt for one of our categories, that is
  recorded as such — no prompt is invented. This is the realistic case for every one of MVTec AD 2's
  eight categories: SAA+'s published prompts cover MVTec AD (classic) and VisA objects, none of which
  are `can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`, `wallplugs` or `walnuts`. Where a
  category has no published prompt, SAA+ runs on the repo's own generic fallback prompt for that
  category instead, and every table marks that category's numbers as *SAA+ without its per-object
  prompts* — never reported as SAA+ proper. If every one of the eight categories falls back this way,
  the paper records that this provision had no effect for MVTec AD 2 and says so plainly, rather than
  presenting fallback numbers as the method its prompts describe. `configs/methods/saa.yaml`'s
  `prompt_coverage` field records, once Colab phase A.3 has read the repo, how many of the eight
  categories actually got a published prompt.

- **Mask-based anomaly maps (SAA+).** SAA+ emits region masks with confidence scores rather than a
  dense per-pixel field, so its anomaly map has few distinct levels, and every threshold-sweeping pixel
  metric is sensitive to that. The map is taken **verbatim from the repo's own inference output** and
  only upsampled: recomposing it from masks would be a re-implementation (priority 3), and smoothing it
  would be a post-process no other method in this study receives. Map granularity is instead measured
  (`postprocess.distinct_levels`) and reported alongside SAA+'s pixel metrics as a stated confound.

- **Two pinned checkpoints for SAA+.** GroundingDINO and SAM are independent artefacts with independent
  versions; both are pinned by sha256 in configs/methods/saa.yaml. Changing either moves every pixel
  metric without any code change.

### 3.1 Auxiliary-training overlap

A method is **auxiliary-trained** when it learns something (prompts, adapters, a projection) on an
anomaly-detection dataset other than the target, then scores the target zero-shot. Such a method's
numbers are credible only if its auxiliary training data does not overlap the test set. This is a
common credibility hole in the zero-shot AD literature, and closing it is a requirement of this
protocol, not a courtesy:

- Every auxiliary-trained method gets a committed **overlap audit** naming the auxiliary dataset
  behind each checkpoint it uses, the test set each checkpoint is used on, and why that pairing has no
  overlap. The audit is written **before** any number for that method exists.
- Where a method ships several checkpoints, which checkpoint is used for which test set is a
  pre-registration decision recorded in the audit and in the method's config — never a choice made
  after seeing results.
- Overlap is judged at the dataset level first. Where a pairing is clean at the image level but the
  auxiliary data shares a provider or an imaging domain with the test set, that **domain proximity**
  is either avoided by choosing another checkpoint or recorded as an explicit caveat on every table
  where the method appears. It is never reported as clean.
- **Training-free methods** (no method-trained weights at all) still appear in the audit table, marked
  as such. A table with methods silently missing from it reads as an omission, not an exemption.

Audits on file: `docs/anomalyclip-overlap-audit.md`, `docs/adaclip-overlap-audit.md`. SAA+ is
training-free and carries the corresponding table row rather than an audit document.

The paper reports this material in its own §3.1, which is why both audit documents and the method
bullets above cross-reference "§3.1" — the requirement lives here, the reporting lives there.

## 4. Metrics

- Image level: I-AUROC, I-AP, I-F1max.
- Pixel level: P-AUROC, AU-PRO@0.3, AU-PRO@0.05, and the MVTec AD 2 official metric set.
- Efficiency: median and p95 latency per image at batch size 1 on the fixed hardware below,
  parameter count, peak VRAM. API-based baselines report tokens + cost instead of VRAM.
- The official MVTec AD 2 metric set is adopted as the evaluation server defines it. *(Exact metric
  names and definitions are read from the server's submission documentation at download time.)*
- **Evaluation resolution.** Pixel metrics are computed at the dataset's native resolution
  (1400x1900 for Vial), against unmodified ground-truth masks. Methods may run inference at
  whatever internal resolution they were designed for, but every adapter returns its anomaly map
  upsampled to native resolution; nothing downsamples a mask. Downsampling masks would be cheaper,
  but it shrinks small defect regions and AU-PRO weights every connected region equally — a tiny
  defect can vanish entirely and take its equal share of the score with it. Since AU-PRO is the
  metric this benchmark rests on, that cost is not acceptable to save memory.
- **Aggregation granularity.** Pixel metrics are aggregated **per lighting condition**, not over a
  whole category at once. This is what the study wants scientifically — MVTec AD 2 exists to
  measure robustness to lighting shift — and it is also what makes native-resolution evaluation
  possible on the reference platform: one condition is 20 images (about 53 Mpx, ~3.4 GB peak),
  where a whole category is 140 images (372 Mpx, ~23.8 GB) and does not fit. Per-category means are
  composed from the per-condition results rather than computed in one pass. `pixel_metrics`
  enforces this with a memory budget that refuses the whole-category case.

- **MLLM response parsing.** The Qwen baseline returns free text. A response is parsed for the
  required JSON (`anomaly_probability` plus a `cells` list on the 7x7 grid); prose around the JSON
  and an out-of-range probability are tolerated, invalid cell labels are dropped. A response that
  cannot be parsed carries no information, so its image score is 0.5 (the no-information point for
  AUROC), never a silent 0 and never NaN, and the failure is recorded per sample
  (`extras.parse_ok = False`) so the parse-failure rate is a reported number. This policy is fixed
  for every dataset and category, like the prompt itself.

- **Anomaly-map scale.** Pixel metrics (P-AUROC, AU-PRO, SegF1) rank pooled pixels across every
  image in a group, so a map's *absolute* scale does not matter but its scale must be *consistent
  across images*. Methods therefore return maps on their own internal scale (e.g. PatchCore's raw
  patch distances), NOT rescaled per image to [0,1]. Per-image [0,1] normalisation would stretch a
  clean image's map to the same range as a defective one and destroy the cross-image ranking the
  metrics depend on; it is used only for visualisation, never for scored maps.

## 5. Hardware & software

Accuracy metrics are computed on Google Colab (free tier, single T4, fp16). The assigned GPU varies
between sessions, so **no latency number is ever reported from a Colab session**. The efficiency
table is measured in M5 in a single session on one rented fixed instance, using the official MVTec
AD 2 runtime and memory-footprint utilities, documented in `results/ENVIRONMENT.md` with driver and
library versions. Every accuracy row records the GPU Colab assigned, so runs stay auditable despite
varying hardware.

## 6. Statistical hygiene

Three seeds where any stochasticity exists; report mean ± std. No cherry-picking categories:
every table reports all categories or states explicitly why one is excluded (e.g., official
protocol exclusion). Failed runs are reported as failures, not silently dropped.

## 7. What we will NOT do

- No per-dataset prompt tuning after seeing test results.
- No "best of N runs".
- No mixing of our PatchCore anchor numbers with published numbers in the same column without marking.
- **No iteration against the evaluation server.** One submission per method, final, made only after
  that method's configuration is frozen. The private test split has hidden ground truth; repeated
  submissions with selection of the best outcome is the same "best of N" forbidden above. Submission
  date and returned scores are recorded in `results/`.

## Changelog

- 2026-07-11 — v0.1 initial frozen draft.
- 2026-07-22 — v0.2. Scope corrected after verifying dataset facts that v0.1 assumed:
  Real-IAD requires no application and is 622 GB (removed from scope); MVTec AD 2 private test
  ground truth is evaluation-server only (§2, §4); Colab has no fixed hardware (§5); added the
  one-submission-per-method rule (§7). MLLM baseline pinned to Qwen2.5-VL-3B-Instruct to fit 16 GB.
  Rationale: docs/superpowers/specs/2026-07-22-scope-v0.2-colab-design.md
- 2026-07-23 — v0.2.1. Verified against the official download page, resolving two of the three
  items §2 left open: the five split directory names, and the 30.4 GB download size (available
  per category, largest 10 GB). Records that `test_public` spans every lighting condition with
  pixel ground truth, so the lighting-shift analysis does not depend on evaluation-server
  access. Official metric names remain unverified pending the server's submission
  documentation. No evaluation rule changed.
- 2026-07-23 — v0.2.2. Layout verified against the real archive rather than the download page:
  five splits with the subdirectory structure recorded in docs/datasets-access.md, lighting
  condition encoded in every filename, masks named `{stem}_mask.png`. Adds two preprocessing
  rules that touch every number: grayscale images are converted to 3-channel RGB (§3), and
  withheld-label samples are marked -1 and excluded from accuracy metrics (§2). Official metric
  names still unverified. No evaluation rule changed.
- 2026-07-23 — v0.2.3. Documentation only. Records the float16 anomaly-map storage precision
  in §3, with the measured effect on every pixel metric (worst case 8.0e-04, on SegF1max),
  because it is the same class of decision as the RGB conversion already recorded there: a
  preprocessing choice that touches every reported pixel-level number. The downcast itself is
  unchanged and predates this entry. No evaluation rule changed.
- 2026-07-23 — v0.2.4. Records the evaluation resolution and aggregation granularity in §4, both
  decided before any result was computed. Pixel metrics run at native resolution against
  unmodified masks (adapters upsample; masks are never downsampled, because shrinking regions
  distorts AU-PRO, which weights every region equally). Aggregation is per lighting condition,
  which the study wants anyway and which is what makes native resolution fit the reference
  platform's memory. This constrains how numbers are produced, so unlike v0.2.1-v0.2.3 it is not
  documentation-only.
- 2026-07-24 — v0.2.5. Records the MLLM response-parsing and parse-failure policy in §3 (score 0.5
  and a per-sample flag on an unparseable response, so parse failures are reported rather than read
  as confident normals). Fixes it before any MLLM number is produced. No other evaluation rule changed.
- 2026-07-24 — v0.2.6. Two contract changes, both forced by the first full-shot method (PatchCore).
  §3: full-shot anchors fit a per-category memory bank on the train split (`fit()` on the method
  contract, called per category by the runner). §4: anomaly maps are on the method's own consistent
  scale, not per-image normalised to [0,1] — per-image rescaling breaks the cross-image pixel-metric
  ranking. This is an evaluation-affecting change (it alters the pixel numbers a per-image-normalised
  method would have produced), decided before any full-shot or real anomalib number exists.
- 2026-07-25 — v0.2.7. Records AnomalyCLIP's auxiliary-training regime and the checkpoint-per-test-
  set overlap audit (§3, docs/anomalyclip-overlap-audit.md): VisA-trained checkpoint for MVTec AD 2
  (conservative against MVTec-family domain proximity), MVTec-AD-trained for the VisA reproduction.
  Decided before any AnomalyCLIP number exists. No evaluation rule for other methods changed.
- 2026-07-29 — v0.2.8. Records AdaCLIP's auxiliary-training regime and its checkpoint-per-test-set
  overlap audit (§3, docs/adaclip-overlap-audit.md), on the same rule already fixed for AnomalyCLIP:
  VisA-trained checkpoint for MVTec AD 2, MVTec-AD-trained for the VisA reproduction. Also pre-registers
  the single-checkpoint contingency — if the repo publishes only one, it is used and the domain
  proximity is carried as a caveat on every table rather than reported as clean. Decided before any
  AdaCLIP number exists. No evaluation rule for other methods changed.
- 2026-07-30 — editorial, **still v0.2.8** (no rule changed, so no version bump; v0.2.9 stays reserved
  for the pending SAA+ amendment). Adds the §3.1 subsection this document had been cross-referencing
  since v0.2.7 without containing: §3 and both overlap-audit documents cited "§3.1" as the
  auxiliary-training credibility requirement, but the protocol's headings ran §1–§7 with no §3.1, so
  every such reference resolved to nothing. The new subsection states the requirement that was already
  being applied — per-checkpoint overlap audits written before any number exists, checkpoint choice
  pre-registered, domain proximity avoided or carried as an explicit caveat, training-free methods
  listed rather than omitted — and records that the paper reports this material in its own §3.1. It
  introduces no new obligation and changes no method's treatment.
- 2026-07-30 — v0.2.9. Clarifies §3's governing principle (verbatim-sourced per-category content is
  reproduction, not "engineering") and applies it to SAA+, whose per-category prompts are published in
  its repo rather than a paper; also states SAA+'s conditions. (a) Prompts: SAA+'s per-object domain
  prompts are its contribution, taken verbatim from the official repo at the pinned commit, recorded
  with file provenance in configs/methods/saa_prompts.yaml, committed before the first scoring run, and
  never changed after seeing a result; unpublished prompts are recorded as unpublished, never invented.
  Where a category has no published prompt — the realistic outcome for all eight MVTec AD 2 categories,
  since SAA+'s published prompts cover MVTec AD (classic) and VisA objects only — SAA+ runs on the
  repo's own generic fallback prompt for that category, and every table marks those numbers as *SAA+
  without its per-object prompts*, never as SAA+ proper; if that applies to every category, the paper
  records that this provision had no effect on MVTec AD 2. (b) Map provenance: SAA+'s mask-based map is
  taken verbatim from the repo and only upsampled — never recomposed, never smoothed — with its
  granularity measured and reported as a confound. (c) Two checkpoints (GroundingDINO, SAM) are pinned
  by sha256, not one. Decided before any SAA+ number exists. No evaluation rule for other methods
  changed.
