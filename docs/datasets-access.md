# Dataset access

## MVTec AD 2 — primary benchmark

1. Download from the official MVTec research page (registration + CC BY-NC-SA 4.0 research license).
2. Register for the evaluation server at https://benchmark.mvtec.com/ — the private test split is
   scored only there.
3. Download the official PyTorch dataset class and utils bundle from the same page. It carries the
   authoritative split names, the submission format, and runtime/memory measurement helpers.

Record on download (these are unverified until then, by design):

- Download size: ____
- Dataset version / date: ____
- Split directory names, read from the official dataset class: ____
- Official metric names, read from the server submission docs: ____

### ⚠️ Open blocker — evaluation-server registration requires a company email address

Observed on the benchmark login page (2026-07-23): registration asks for a username that is an
email address, and states that **a company email address is required**. Free providers
(gmail.com and similar) are therefore expected to be rejected. The page directs registration
problems to `webmaster@mvtec.com`.

This is a hard dependency, not a nuisance: the private test split is scored *only* on that
server, so without an account the protocol's §2 private-split reporting, its §4 official metric
set, and the §7 one-submission-per-method rule are all unexecutable, and M4 cannot happen.

Routes being tried, in order:

1. An institutional address (University of Luxembourg), if still active.
2. A request to `webmaster@mvtec.com` explaining the independent academic research context.

Status: ☐ not resolved — route 2 request sent 2026-07-23, awaiting reply.
If no response by **2026-08-06** (two weeks), follow up once; if that also goes unanswered,
treat the private split as unavailable and open the protocol amendment described below rather
than letting M4 sit blocked indefinitely.

- Address the account was granted to: ____
- Registered on: ____

**If both routes fail**, the study is limited to the public test split. That is a second scope
reduction and must be handled the same way as the first: a dated Changelog entry in
`docs/protocol.md` amending §2/§4/§7, plus corresponding changes to the paper's contributions —
not a quiet omission. Note that this affects only the *evaluation server*; the dataset download
itself is a separate registration under the CC BY-NC-SA 4.0 research license, and local metrics
on the public split remain fully available either way.

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
