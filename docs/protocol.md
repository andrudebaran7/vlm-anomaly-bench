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
from the original papers. The MLLM baseline uses one fixed structured prompt + one fixed scoring
rubric for all datasets, published in `configs/methods/mllm_qwen.yaml` — no per-category prompt
engineering.

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
