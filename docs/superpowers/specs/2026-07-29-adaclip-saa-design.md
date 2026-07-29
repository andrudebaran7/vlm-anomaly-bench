# AdaCLIP and SAA+ adapter design (spec)

**Date:** 2026-07-29
**Status:** approved, pre-implementation
**Produces:** two implementation plans — `docs/superpowers/plans/` — one per method, executed independently.

AdaCLIP and SAA+ are the last two methods in `docs/next-steps.md` (steps 4 and 5) with no plan
written. Both adapter modules are still stubs. This spec fixes the design decisions that must be made
*before* any number exists, so the Colab sessions execute a decision rather than improvise one.

## Why one spec and two plans

Both methods share the architecture already validated three times in this repo (WinCLIP, AnomalyCLIP,
PatchCore), so the architecture is written once, here. They are split into two plan files because they
execute in separate Colab sessions and carry unrelated risks: AdaCLIP is auxiliary-trained and needs an
overlap audit, SAA+ is training-free but is the heaviest and slowest run in the study. A combined plan
would couple two independent GPU sessions, so an SAA+ abort would strand AdaCLIP's checkboxes.

## Shared architecture

- CPU adapter in `src/vlmab/methods/{adaclip,saa}.py` over an **injectable backend**. Both declare
  `zero_shot = True`. Without a backend, `prepare()` and `predict()` raise `MethodNotRunnable`, so the
  CLI never pretends a method ran.
- **Seam, identical in both:** `score(image: np.ndarray, category: str) -> tuple[float, np.ndarray]`.
  For SAA+ the category is substantive (it selects the domain prompts). For AdaCLIP it is the superset
  choice — see "The seam shape" below.
- The adapter only calls `upsample_to(map, image.shape[:2])` and passes the score through raw. No
  per-image normalisation (protocol v0.2.6).
- CPU tests use a fake backend exclusively. Neither the tests nor the modules may import torch, the
  official repos, transformers or open_clip at load time — CI installs only
  `numpy scipy scikit-learn pandas pyarrow pillow pytest`.
- **Provenance for both: official repo at a pinned commit** (protocol §3 priority 1 — anomalib ships
  neither).

### The seam shape

WinCLIP's seam takes a category (the object noun goes in the prompt); AnomalyCLIP's does not (its
prompts are object-agnostic). AdaCLIP uses hybrid learnable prompts — a static learned component and a
per-image dynamic component — and whether its inference path consumes a class name **cannot be verified
without the repo in front of you**.

**Decision: the seam takes a category in both methods.** If AdaCLIP turns out to be object-agnostic, the
backend ignores the argument and says so in its docstring — no change to the adapter, the tests or the
plan. If it does consume the class name, the plan already fits. The alternative (mirroring
AnomalyCLIP's category-free seam) risks discovering the mismatch mid-GPU-session, which is the most
expensive moment to rewrite an adapter, its tests and a plan. `AnomalyMethod.predict()` already receives
`category` unconditionally, so an inert argument is a case the contract covers.

## AdaCLIP

AdaCLIP (Cao et al., ECCV 2024) adapts CLIP with hybrid learnable prompts and is **auxiliary-trained**,
so it inherits AnomalyCLIP's whole credibility apparatus: its numbers are valid only if the auxiliary
training data does not overlap the test set (protocol §1, §3.1).

### Pre-registered checkpoint choice

Deliberately identical to AnomalyCLIP's. Both auxiliary-trained methods following the same rule is what
makes the paper's §3.1 table legible.

| Test set | Checkpoint | Why it is clean |
|---|---|---|
| MVTec AD 2 (primary) | VisA-trained | Conservative: avoids domain proximity with the MVTec family. |
| VisA (§2 gate) | MVTec-AD-trained | Clean: different datasets. |

### Contingency, pre-registered

The table above assumes AdaCLIP publishes both checkpoints. That is unverified. If only the
MVTec-AD-trained checkpoint is published, the primary run falls into exactly the domain-proximity case
avoided for AnomalyCLIP. The rule, written before any number exists:

> If only the MVTec-AD-trained checkpoint exists, use it for MVTec AD 2 **and** record the domain
> proximity explicitly in `docs/adaclip-overlap-audit.md` and as a footnote in every table where
> AdaCLIP appears. Do not use it as though it were clean, and do not drop the method silently.

This mirrors how `docs/datasets-access.md` handles the evaluation-server blocker: the decision rule is
written down before the outcome is known.

### Deliverables

CPU (subagent-executable, TDD):
1. `AdaClipRef` adapter + `tests/test_adaclip.py` (fake backend) + pre-registered
   `configs/methods/adaclip.yaml`.
2. Register `adaclip`; re-point the registry's unregistered example (see "The registry chain").
3. `docs/adaclip-overlap-audit.md` + protocol §3 amendment and Changelog entry → **v0.2.8**.

Colab (interactive, GPU): Phase A environment and pinned commit; Phase B backend derived from the
official `test.py` with VERIFY steps; Phase C end-to-end on Vial; Phase D VisA ±1.0 gate with the
MVTec-AD-trained checkpoint.

## SAA+

SAA+ is a **training-free** cascade: GroundingDINO proposes regions from defect language prompts, SAM
refines them into masks. No method-trained weights — two frozen foundation models.

No overlap audit is required. The plan still adds SAA+'s row to the paper's §3.1 table, stating
*training-free, no auxiliary data*: an overlap table with methods missing from it reads as an omission,
not an exemption.

### 1. Per-category prompts — protocol §3 amendment (v0.2.9)

Protocol §3 states prompts are taken verbatim from the original papers with *no per-category prompt
engineering*. SAA+'s central contribution — hybrid prompt regularization — is per-object domain
knowledge: defect language expressions and object-specific area/size constraints. Running SAA+ without
them is not SAA+; running it with them violates the letter of §3.

**Decision: a dated §3 amendment that exempts SAA+ explicitly.** Its per-object prompts are taken
**verbatim from the official repo at the pinned commit**, never tuned by us and never adjusted after
seeing a result. The plan pre-registers the full per-category prompt table **as a committed file before
anything runs**, so the amendment is verifiable rather than a promise.

Rejected: running SAA+ with a single generic prompt (complies with §3 but measures a degraded method its
own paper never reports, and would almost certainly fail the VisA gate); reporting both variants (most
informative, but doubles the compute of the study's slowest method).

### 2. Two checkpoints to pin, not one

GroundingDINO and SAM are independent artefacts with independent versions. Both get a sha256 recorded in
`configs/methods/saa.yaml`. This is a classic silent-failure mode: swapping the SAM checkpoint moves
every pixel metric without touching a line of code.

### 3. The anomaly map is not dense

SAA+ produces region masks with confidence scores, not a continuous per-pixel field. Our pixel metrics
(P-AUROC, AU-PRO@0.3/@0.05, SegF1) sweep thresholds, and a map composed of a handful of masks has very
few distinct levels — which depresses AU-PRO in a way that reflects the output format as much as real
localisation quality.

**Decision: take the repo's own final anomaly map verbatim**, at the pinned commit, without recomposing
it. Any composition of our own would be a re-implementation (protocol §3 priority 3, flagged in every
table) and would risk failing the VisA gate on our composition rather than on the method. The plan adds
a **granularity diagnostic** — record how many distinct levels a typical map has — and a stated
limitation in the paper's §5.2 if that count turns out to be low.

Rejected: compositing masks by max confidence (re-implementation); Gaussian-smoothing the official map
(a post-process no other method receives, breaking the comparability the protocol protects).

### 4. Compute — the declared risk is smaller than it looks

The protocol flags SAA+ as the run that may not fit a 12h Colab session. Existing infrastructure already
addresses this: the runner is resumable and the store is crash-safe, with the category as simultaneously
the download, resume and shard unit. SAA+ does not need to fit one session; it needs to fit N resumable
ones.

What is missing, and what the plan adds, is a **cost probe in Phase C**: measure seconds/image over ~10
Vial images on a T4, extrapolate to the whole category and to the full 8-category grid, and record all
three figures *before* committing sessions. With the measured figure in hand the continue/abort decision
is made on data; without it, it is discovered nine hours in. The recorded extrapolation — not a guess —
is what a later decision to reduce SAA+'s scope must cite, and if SAA+ ends up run on fewer than eight
categories that is reported as a stated limitation with the measured cost attached, per protocol §6:
report the failure as a failure.

### 5. Gate-metric contingency, pre-registered

The §2 gate is defined as *published image-AUROC ±1.0*. SAA+ is a segmentation method whose paper
emphasises pixel metrics, and may not publish VisA image-AUROC. The rule, written before looking:

> If SAA+'s paper does not publish VisA image-AUROC, the gate runs on the first of these the paper
> does publish, in this order: **I-AP, then I-F1max** — both already in
> `src/vlmab/metrics/image_level.py`. `results/reproduction/saa_visa.md` records **explicitly which
> metric the gate ran on**. The ±1.0 tolerance does not change, and the gate is not replaced by a
> qualitative comparison. If the paper publishes no image-level metric on VisA at all, SAA+ is reported
> as ungated in every table (protocol §2), not silently exempted.

### Deliverables

CPU: `SaaRef` adapter + tests (category substantive in the seam); register `saa` + re-point the registry
example; pre-registered per-category prompt table + protocol §3 amendment → **v0.2.9**.

Colab: Phase A environment with two pinned repos/checkpoints; Phase B backend over the official
inference path with VERIFY steps, including the map granularity diagnostic; Phase C Vial end-to-end plus
the cost probe; Phase D VisA gate.

## The registry chain

`tests/test_registry.py::test_unknown_method_lists_the_known_ones` currently calls
`build_method("adaclip")` as its unregistered example — the AnomalyCLIP plan left it there deliberately,
with a comment. Registering AdaCLIP invalidates it, so the AdaCLIP plan re-points it to `saa`; the SAA+
plan must then move it again when it registers `saa`. Both plans state this so it is not discovered as a
CI failure.

## Verification

Two regimes, kept separate:

- **CPU half — real, automatable verification.** Strict TDD: failing test, implementation, passing test,
  full suite. The bar is that the existing suite (223 tests as of 2026-07-29) stays green and that no
  test or module imports torch, the official repos, transformers or open_clip at load time.
- **Colab half — a recipe with VERIFY steps, not verified code.** No playbook asserts what the official
  repo calls; it instructs the executor to read `test.py` at the pinned commit and write down the exact
  calls. Fabricating an unpackaged repo's API is what the protocol forbids. The smoke test and the VisA
  gate are the proof, not pre-written code.

**Gates.** Non-negotiable for both: no MVTec AD 2 number until the VisA reproduction lands within ±1.0.
Each plan carries a "if the gate fails" section listing likely causes in order, and the instruction to
flag the method as unreproduced in every table rather than omitting it.

## Execution order

AdaCLIP first, then SAA+ — the order `docs/next-steps.md` already fixes (steps 4 and 5), and mechanically
cleaner: it chains the registry-example re-pointing (`adaclip` → `saa` → next) and reserves the protocol
amendments in sequence, v0.2.8 for AdaCLIP and v0.2.9 for SAA+.

## Out of scope

The M3 full grid; the M4 evaluation server; any change to the four already-planned methods. These plans
do not front-run the PatchCore Colab session, which remains step 1.
