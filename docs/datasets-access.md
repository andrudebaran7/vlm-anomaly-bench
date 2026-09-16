# Dataset access

## MVTec AD 2 — primary benchmark

1. Download from the official MVTec research page (registration + CC BY-NC-SA 4.0 research license).
2. Register for the evaluation server at https://benchmark.mvtec.com/ — the private test split is
   scored only there.
3. Download the official PyTorch dataset class and utils bundle from the same page. It carries the
   authoritative split names, the submission format, and runtime/memory measurement helpers.

Verified from the official download page, 2026-07-23:

- **Download size: 30.4 GB** total. Also available per object category, which changes how it is
  used on Colab — see below.
- **Split directory names** (five folders per object):
  - `train` — defect-free training images
  - `validation` — defect-free validation images
  - `test_public` — test images from **every lighting condition**, with pixel-precise ground
    truth annotations
  - `test_private` — test images under the regular lighting condition, ground truth withheld
  - `test_private_mixed` — the **same scenes as `test_private`** under varied lighting, ground
    truth withheld
- Dataset version / date: ____ (record on download)
- Official metric: pixel-level **SegF1**, pooled over all test pixels rather than averaged per image
  — verified 2026-07-30 from the VAND 3.0 challenge report, recorded in protocol §4 (v0.2.10). The
  full definition and the submission mechanics are in the evaluation-server section below.
- Dataset paper (verified 2026-07-30): Heckler-Kram, Neudeck, Scheler, König and Steger, *The MVTec
  AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection*, arXiv:2503.21622; also IJCV
  vol. 134. Released for the VAND 3.0 challenge at CVPR 2025. Reports that state-of-the-art methods
  stay **below 60% average AU-PRO** on this dataset.

### Per-category download sizes

| Category | Size |
|---|---|
| Fabric | 10 GB |
| Rice | 6.29 GB |
| Walnuts | 5.88 GB |
| Can | 2.65 GB |
| Wallplugs | 2.05 GB |
| Sheet Metal | 1.53 GB |
| Fruit Jelly | 1.2 GB |
| Vial | 0.77 GB |

### Verified directory layout (read from `vial.tar.gz`, 2026-07-23)

    <root>/<category>/
      train/good/                  defect-free, lighting: regular only
      validation/good/             defect-free, lighting: regular only
      test_public/
        good/                      normal, all 7 lighting conditions
        bad/                       anomalous, all 7 lighting conditions
        ground_truth/bad/          one mask per bad image
      test_private/                flat, regular only, labels withheld
      test_private_mixed/          flat, mixed only, labels withheld

- Image filename: `{index:03d}_{condition}.png` (e.g. `000_shift_1.png`).
- Mask filename: `{index:03d}_{condition}_mask.png`.
- Lighting conditions: `regular`, `overexposed`, `underexposed`, `shift_1`..`shift_4`, and
  `mixed` in `test_private_mixed`.
- Vial counts: train 291, validation 41, test_public 35 good + 105 bad (+105 masks),
  test_private 276, test_private_mixed 276.
- **Vial images are 1400x1900, 8-bit grayscale (2.66 MP)** — not RGB. Verified for Vial only;
  record per category as each is downloaded, since the loader must handle both.
- `test_public/good/` and `test_public/bad/` reuse the same stems (`000_regular.png` exists in
  both), so anomaly-map filenames must be derived from more than the stem.
- The bundled `readme.txt` carries attribution and the CC BY-NC-SA 4.0 licence only. It does
  **not** document the official metric definitions — those remain behind the evaluation server.

Per-category image format (record on download):

| Category | Resolution | Channels |
|---|---|---|
| Vial | 1400x1900 | 8-bit grayscale |
| Can | ____ | ____ |
| Fabric | ____ | ____ |
| Fruit Jelly | ____ | ____ |
| Rice | ____ | ____ |
| Sheet Metal | ____ | ____ |
| Wallplugs | ____ | ____ |
| Walnuts | ____ | ____ |

**This is why per-category downloads matter.** 30.4 GB does not fit Google Drive's free 15 GB
tier, and re-fetching it whole every session is the dominant cost on a platform that
disconnects. Fetching one category, evaluating every method on it, then discarding it caps peak
disk at the largest single category — **10 GB, not 30.4 GB** — and makes a session
self-contained. This lines up exactly with the runner's per-category checkpoint granularity
(`src/vlmab/eval/runner.py`): one category is simultaneously the download unit, the resume unit
and the shard unit.

### How a category reaches a Colab session (verified 2026-09-16)

Via **Google Drive**, uploaded once per category, not re-downloaded per session: the archives sit
behind mvtec.com's registration form, which an unattended session cannot clear. The convention the
notebooks use is `MyDrive/mvtec_ad2/<category>.tar.gz`, holding the unmodified archive from the
download page.

Layout inside the archive, read from `vial.tar.gz`: the top level is `<category>/`, `license.txt`
and `readme.txt`, so `tar -xzf <tar> -C <root>` lands the category at `<root>/<category>` with no
path surgery. Extract into the clone's `data/` — it is gitignored, so `git reset --hard` in the
notebook's sync cell leaves it alone — and **never into `/tmp`**, which is tmpfs, i.e. RAM.

One category at a time is what keeps this inside Drive's free 15 GB tier, the same reason the
per-category split matters above. Verified end to end for Vial: `scripts/prepare_data.py --root
data/mvtec_ad2 --category vial` prints the counts recorded above and exits 0.

### The lighting-shift analysis does NOT depend on the evaluation server

`test_public` contains images from **every lighting condition together with pixel ground
truth**. So the lighting-robustness question — the reason MVTec AD 2 exists and the core of the
paper's §5.1 and Figure 2 — is answerable entirely offline.

What the private split adds is narrower and more specific: `test_private` and
`test_private_mixed` are the *same scenes* under regular versus varied lighting, i.e. a
controlled paired comparison that removes scene as a confounder. That is a real strengthening,
not the whole story.

This materially reduces the damage of the evaluation-server blocker below: without server
access the study loses the leaderboard entry and the paired same-scene contrast, but keeps the
lighting-shift result, all pixel-level localisation metrics, and the full method comparison.

### ✅ RESOLVED — evaluation-server access granted (2026-07-30)

**Status: access authorised 2026-07-30.** M4 is unblocked and the private test split is in scope:
protocol §2 private-split reporting, §4's official metric set and §7's one-submission-per-method
rule are all executable. The scope-reduction amendment drafted below was **never needed** and is
kept only as the record of what the fallback would have been.

To record now that access exists (fill on first login):

- Address the account was granted to: ____
- Registered on: ____

**Metric and submission mechanics — answered 2026-07-30 from a secondary but authoritative source**,
the VAND 3.0 challenge report (arXiv:2503.21622's companion, arXiv:2509.17615 §4.2–§4.5), written by
the MVTec team that runs this server and documenting its Category 1 evaluation. Recorded in protocol
§4/§7 (v0.2.10). **Still confirm each against the server's own submission docs on first login** — a
challenge report describes one edition's rules, and the server is the authority:

- Official metric: pixel-level **SegF1**, precision and recall pooled over the complete set of test
  pixels, **not** averaged per image; per-category scores averaged over the eight categories; rank =
  mean of the ranks on `test_private` and `test_private_mixed`. Confirmed against server docs: ☐
- Submission payload: **both** thresholded and continuous anomaly maps per test split. This is what
  forces a ground-truth-free threshold choice, which the repo now makes via the pre-registered
  threshold rule (protocol §4, v0.2.11 — was a hard prerequisite for M4). Confirmed against server docs: ☐
- Rate limit: **two submissions per week per account.** Looser than §7's one-per-method rule, so §7
  stays binding. Confirmed against server docs: ☐
- Anything the server's docs say that the above does not cover: ____

#### History of the blocker (kept — it explains the protocol's contingency language)

Observed on the benchmark login page (2026-07-23): registration asks for a username that is an
email address, and states that **a company email address is required**. Free providers
(gmail.com and similar) were therefore expected to be rejected. The page directs registration
problems to `webmaster@mvtec.com`.

This was a hard dependency, not a nuisance: the private test split is scored *only* on that
server, so without an account M4 could not happen.

Routes tried, in order:

1. An institutional address (University of Luxembourg), if still active.
2. A request to `webmaster@mvtec.com` explaining the independent academic research context.

Route 2 request was sent 2026-07-23 and **granted 2026-07-30**, seven days later — before the
2026-08-06 follow-up deadline the decision rule had set, so no follow-up was sent and no scope
reduction was opened.

**Had both routes failed**, the study would have been limited to the public test split — a second
scope reduction, handled like the first: a dated Changelog entry in `docs/protocol.md` amending
§2/§4/§7, plus corresponding changes to the paper's contributions, not a quiet omission. Note that
this only ever affected the *evaluation server*; the dataset download is a separate registration
under the CC BY-NC-SA 4.0 research license, and local metrics on the public split were available
either way.

## VisA + MVTec AD (classic) — sanity only

Used solely to validate our implementations against published numbers (±1.0 I-AUROC,
protocol §2). VisA stays in this role deliberately: it is the least-friction dataset here, but
published methods already score high on it, so promoting it to a primary benchmark would
reintroduce exactly the saturation this study exists to escape.

**VisA** — verified 2026-07-23:

- Size: **1.80 GiB** (`VisA_20220922.tar`, 1,929,840,640 bytes, measured by HTTP HEAD).
- License: **CC BY 4.0** — the most permissive of any dataset here (attribution only, no
  NonCommercial clause).
- Access: AWS Open Data Registry, **no AWS account and no form**:
  `aws s3 cp --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar .`
  (bucket `amazon-visual-anomaly`, region `us-west-2`; published 2022-09-22).
- Content: 10,821 images (9,621 normal, 1,200 anomalous), 12 classes across 3 domains.

### Verified directory layout (read from `VisA_20220922.tar`, 2026-09-16)

Downloaded over plain HTTPS — no AWS CLI and no account needed, the `s3 cp` line above is one
route of two:

    https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar

Size on download: **1,929,840,640 bytes**, byte-exact with the HTTP HEAD recorded 2026-07-23.

    <root>/split_csv/1cls.csv        the one-class protocol  <- the authority
    <root>/split_csv/2cls_highshot.csv, 2cls_fewshot.csv     the two-class protocols
    <root>/<object>/Data/Images/Normal/0000.JPG
    <root>/<object>/Data/Images/Anomaly/000.JPG
    <root>/<object>/Data/Masks/Anomaly/000.png
    <root>/<object>/image_anno.csv   per-object annotations, not needed by the loader
    <root>/LICENSE-DATASET

- `1cls.csv` columns: `object,split,label,image,mask`. `split` is `train` or `test`; `label` is
  `normal` or `anomaly`; `image` and `mask` are paths relative to the root. **The mask field is
  empty on every normal row** and a real path on every anomalous one.
- Images are uppercase `.JPG`, masks lowercase `.png`.
- **Counts read from the CSV: 8,659 train normal; 962 test normal; 1,200 test anomalous;
  12 objects.** Total 10,821 and 9,621 normal, which reproduces the paper's stated figures, and
  8,659/9,621 = 90.0%, which reproduces its one-class protocol ("assigning 90% normal images to
  train set while 10% normal images and all anomalous samples are grouped as test set").
- Objects: candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3,
  pcb4, pipe_fryum.
- Sample resolution (candle): 1168x1284 RGB. Not uniform across objects — record per object if
  one ever matters.
- **The splits exist only in that CSV.** The directory tree cannot reconstruct the 90/10 normal
  split, so `split_csv/1cls.csv` is not optional and the loader raises naming it when absent.
- There is **no validation split.** Threshold calibration needs one and runs on MVTec AD 2.
- Loader: `src/vlmab/datasets/visa.py`. `tests/test_visa.py` carries two opt-in tests that run
  only where the archive is on disk and assert the counts above.


**MVTec AD (classic)** — >5,000 images, 15 object and texture categories, from the MVTec
research page (form + license). Download size not published; record it here on download.

## Other freely-accessible AD benchmarks — candidates, not committed

Surveyed 2026-07-23 while the MVTec AD 2 evaluation-server registration was blocked. None of
these is adopted; they are recorded so the option is documented rather than re-researched.

| Dataset | Size | License | Access | Verified |
|---|---|---|---|---|
| MVTec LOCO AD | not published | CC BY-NC-SA 4.0 | form on mvtec.com | license yes, size no |
| KolektorSDD2 | not published | CC BY-NC-SA 4.0 | direct, vicos.si | license yes, size no |
| BTAD | 2,540 images, 3 products | unverified | — | no |
| MPDD | unverified | unverified | github.com/stepanje/MPDD | no |
| DAGM | 10 texture classes | unverified | — | no |

**MVTec LOCO AD is the one worth a second look**, and not as an access fallback. It carries
*logical* anomalies (violations of count and composition constraints) alongside structural
ones — an axis CLIP-based methods are structurally poor at, since their prompts describe
appearance rather than constraint violations, and one where an MLLM baseline has a plausible
path to winning rather than only losing. Before considering it, check whether any zero-shot/VLM
method already reports on it; the contribution claim depends on that being unanswered.

**Not candidates:** DocVQA, TextVQA and VQAv2 were raised and rejected. They are visual
question answering benchmarks with no normal/anomalous labels and no pixel ground truth, so
every image- and pixel-level metric in `src/vlmab/metrics/` is inapplicable to them. Swapping
them in would not narrow this study's scope; it would replace it with a different one.

## Real-IAD / Real-IAD Variety — out of scope

Correction to the v0.1 assumption: **no application is required.** The dataset is publicly
accessible at https://huggingface.co/datasets/Real-IAD/Real-IAD under research-only terms.

It is out of scope for compute reasons, not access reasons: 622 GB for the full release, ~53 GB for
the 1024px variant, and Real-IAD Variety comprises 198,950 images across 160 categories. None fits
the Colab budget. Revisit only with a different compute platform (spec §Scope).

## Auxiliary-training data (AnomalyCLIP / AdaCLIP)

Record exactly which auxiliary dataset each official checkpoint was trained on, and verify zero
overlap with MVTec AD 2. This table goes in the paper (§3.1) — it is a common credibility hole in
this literature.
