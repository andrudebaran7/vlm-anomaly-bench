# Next steps

Living map of where execution stands and what to do next. Milestones (M1–M6) live in the README;
this file is the operational view — which plans are written, which halves are executed, and the
order for the work that remains. **Every method now has a written plan** (2026-07-29: AdaCLIP and
SAA+ were the last two). AdaCLIP's and SAA+'s CPU halves have since landed and are registered. What
remains is either a **Colab/GPU** step or an **external** one. The pre-existing CPU-verifiable core
is done, on `master`, and green on CI (Python 3.11 + 3.13). The **first GPU backend now exists and
runs** — PatchCore's, built in the Colab session of 2026-08-21; see the session note below for what
that cost and what it means for the four methods behind it.

## What is done (CPU, no GPU)

- **Evaluation core:** image/pixel metrics (P-AUROC, AU-PRO, SegF1), provenance, a crash-safe
  result store, a resumable runner with per-sample latency, per-category `fit` for full-shot
  methods, and lighting-grouped aggregation. Protocol frozen at **v0.2.13**.
- **MVTec AD 2 data path:** loader (verified against the real Vial archive), layout verification
  (`scripts/prepare_data.py`), and the run_eval CLI. Native-resolution, raw-scale maps (v0.2.6).
- **Method adapters — CPU halves done and registered** (each wraps an injectable backend; without
  a backend `prepare()` raises `MethodNotRunnable`, so the CLI never pretends it ran):
  - `intensity_baseline` — the dependency-free floor (fully working, no backend needed).
  - `mllm_qwen` — Qwen2.5-VL-3B; the response parser is complete, the model call is the injectable seam.
  - `patchcore_ref` — full-shot anchor; **its anomalib backend is built and fits on a T4**
    (`src/vlmab/methods/patchcore_backend.py`, a tracked file so a Colab cell can pull it; anomalib
    and torch are imported lazily, so CI never touches them). See the session note below.
  - `winclip` — zero-shot; the anomalib WinClip backend is the seam.
  - `anomalyclip` — zero-shot, object-agnostic; the official-repo backend is the seam, plus the
    committed auxiliary-training overlap audit (`docs/anomalyclip-overlap-audit.md`).
  - `adaclip` — zero-shot, hybrid learnable prompts; the official-repo backend is the seam, plus
    the committed auxiliary-training overlap audit (`docs/adaclip-overlap-audit.md`).
  - `saa` — zero-shot, training-free (GroundingDINO + SAM cascade); the official-repo backend is
    the seam, plus the pre-registered per-category prompt table (`configs/methods/saa_prompts.yaml`,
    protocol §3 v0.2.9) and the `distinct_levels` map-granularity diagnostic.

## The gating fact

Everything below needs a **Colab GPU session** (the backends need torch + model weights, which the
dev/CI environment does not have) — except the evaluation-server registration, which is external.
Each Colab playbook is executed **interactively**, not by CPU subagents, and every upstream call in
it has a VERIFY step against the installed version, because fabricating an unverified upstream API is
what the protocol forbids. The acceptance gate for every method is the same: **reproduce its
published VisA image-AUROC within ±1.0 (protocol §2) before any MVTec AD 2 number is reported.**

## Next steps, in order

1. **PatchCore — finish the Colab session** (plan:
   `docs/superpowers/plans/2026-07-24-patchcore-anomalib-backend-colab.md`; driver:
   `notebooks/patchcore_colab.ipynb`, which covers phases 0–2). The full-shot anchor; closes M2
   once its VisA ±1pt gate passes. It is the ceiling every zero-shot number is measured against.
   **Phase 0 and the backend of phase 1 are done** (2026-08-21) — what remains, in order:

   1. ~~**Probe the pre-processing question.**~~ **Answered 2026-08-26, and the answer was the
      unwanted one: `score()` was double-normalising.** `AnomalibModule.forward` runs
      `self.pre_processor` unconditionally, so pre-processing before the call resized and
      ImageNet-normalised every image twice. Fixed — `score()` now hands the model a raw [0,1]
      tensor at native resolution. Probe stage 11 is the standing guard. The `fit` path was never
      affected: anomalib's own pipeline pre-processes the training images exactly once, so the
      memory bank was always built correctly.
   2. ~~**Phase 1.2 smoke test.**~~ **Green 2026-08-26** on the corrected path: normal 16.11 <
      anomalous 69.24, map (256, 256). Probe stage 12 covers the same ground with more of it —
      it drives the shipped backend end to end three times — so phase 1 is closed.
   3. **Phase 2** — Vial end to end through the existing runner (notebook cells 21–26). Cell 22
      now fetches the category from Drive (`MyDrive/mvtec_ad2/<category>.tar.gz`, uploaded once
      per category — `docs/datasets-access.md`); the placeholder is gone, but **the upload itself
      is still a manual prerequisite** and Vial is 0.77 GB. Cell 24 seeds the backend
      (`PatchCoreBackend(seed=0)`) and the runner stamps that seed; phase 2's shard is still a
      plumbing check and must not be reported.
   4. **Phase 3 — RAN 2026-09-16 and the gate FAILED by 0.057. Open, see its section below.**
      Cells 3.1–3.4 in the notebook. **Phase 3b (cells 3b.0–3b.3) is the decided next move**: the
      same 15 categories with `preprocess="classic"`, then the three seeds §6 requires.
      The gate is **MVTec AD classic at 99.0 ± 1.0 I-AUROC**, not VisA — protocol §2 v0.2.12.
      Result committed at `results/reproduction/patchcore_mvtec_ad.md`. The VisA secondary check
      (cell 3.4) was **not run** — the session ended first.
   5. ~~**Phase 4 — freeze provenance.**~~ **Half done 2026-09-17.**
      `configs/methods/patchcore_ref.yaml` now pins `anomalib_version: "2.6.0"` (Colab T4, torch
      2.11.0+cu128, 2026-08-21) instead of the `">=1.1"` range, and a test holds it to an exact
      pin. What remains is mechanical: commit the notebook after the 3b session and confirm both
      CI jobs stay green.
2. **WinCLIP Colab phases (A–D)** — the Colab half of
   `docs/superpowers/plans/2026-07-24-winclip-zeroshot-adapter.md`. First zero-shot method; anomalib
   backend. Watch: if it misses the VisA gate, the likely cause is anomalib's prompt ensemble
   differing from the paper (a §3 priority-2-source risk) — flag it if so.
3. **AnomalyCLIP Colab phases (A–D)** — the Colab half of
   `docs/superpowers/plans/2026-07-25-anomalyclip-zeroshot-adapter.md`. Uses the **official repo**
   (anomalib does not ship it), so its backend is derived from the repo's `test.py` at a pinned
   commit. Use the **VisA-trained** checkpoint for MVTec AD 2 and the **MVTec-AD-trained** checkpoint
   for the VisA reproduction (overlap audit, §3.1).
4. **AdaCLIP Colab phases (A–D)** —
   `docs/superpowers/plans/2026-07-29-adaclip-zeroshot-adapter.md`. Zero-shot and, like AnomalyCLIP,
   **auxiliary-trained**, so it carries the same kind of overlap audit. Its CPU half is done and
   registered (`src/vlmab/methods/adaclip.py`, `docs/adaclip-overlap-audit.md`); what remains is the
   GPU backend. Watch: the plan pre-registers a contingency for the case where the repo publishes
   only one checkpoint — resolve it in Colab phase A.2 before scoring anything.
5. **SAA+ Colab phases (A–D)** —
   `docs/superpowers/plans/2026-07-29-saa-trainingfree-adapter.md`. Training-free (GroundingDINO + SAM
   cascade), so **no aux-training concern** — simpler than AdaCLIP/AnomalyCLIP on the audit side, but it
   carries three things they do not: a dated §3 amendment for its per-category prompts (taken verbatim
   from the repo, committed before any scoring, with `prompt_coverage` in `configs/methods/saa.yaml`
   recording how many of the eight categories actually got one), the repo's own mask-based anomaly map
   taken verbatim plus a granularity diagnostic, and a **measured cost probe** (phase C.2) in place of
   the untested "may not fit a 12h session" assumption — the runner is resumable, so the real question
   is total budget, not session length. Its CPU half is done and registered
   (`src/vlmab/methods/saa.py`, `configs/methods/saa_prompts.yaml`); what remains is the GPU backend and
   phases A–D.
6. **M3 — full MVTec AD 2 grid.** Once each method's VisA gate passes, run the full public-test grid
   over all eight categories, one category at a time (the download/resume/shard unit). Mechanical.
7. **M4 — evaluation server.** Score the private split. Access is **granted** (2026-07-30) and the
   threshold rule is built and pre-registered (v0.2.11); what remains is the submission packaging,
   which needs the server's own format — confirm it on first login. On first login, confirm
   the metric definition, submission payload and attempt limit against the server's own docs — they
   are recorded in `docs/datasets-access.md` from the VAND 3.0 challenge report (a secondary source)
   with a checkbox each.
8. **M5 — efficiency pass** on a rented fixed instance (protocol §5: latency never from Colab).
9. **M6 — preprint.** Paper §1–§3 are already written (see below); §4–§6 need M3.

## The PatchCore Colab sessions (2026-08-21 and 2026-08-26) — what they settled

The first GPU session went entirely into making one backend run. It is worth reading before the
next one, because four of the five methods still ahead go through the same library.

**anomalib 2.6.0 contradicts its own documentation snippets in five ways, each of which surfaced
only after the previous was fixed** (verified on Colab T4, torch 2.11.0+cu128):

1. `Folder(task=...)` — removed in 2.x; the API reference has no such parameter, several snippet
   pages still pass it.
2. `Engine(task=...)` — **accepted** at construction and rejected one call later by Lightning's
   `Trainer`, because the Trainer is built lazily inside `train()`. A swallowed kwarg is worse than
   a rejected one: the traceback points at `fit()`, not at the constructor that took it.
3. `enable_checkpointing=False` — anomalib installs its own `ModelCheckpoint`; Lightning refuses
   the combination.
4. **`enable_progress_bar=False` is required.** Lightning's `RichProgressBar` holds a live display
   while anomalib's coreset sampler writes its own tqdm into it; in a notebook they nest until
   `RecursionError`, with the bar pinned at 0/N. Raising the recursion limit does **not** help
   (tested at 20000) — the nesting is unbounded, not deep-but-finite.
5. `ValSplitMode.NONE` never builds `val_data` while Lightning's fit loop sets up the validation
   loop unconditionally → `AttributeError`. Fixed with `limit_val_batches=0, num_sanity_val_steps=0`.

**The verified Engine configuration is exactly this, and nothing else** — `accelerator`, `devices`
and `max_epochs` were dropped because they were not in the tested set:
`Engine(logger=False, enable_progress_bar=False, limit_val_batches=0, num_sanity_val_steps=0)`.

Two more outside the Engine: the pre-processor carries **no ToTensor**, so `score()` hands it a CHW
float tensor in [0,1], not a PIL image; and Lightning returns the model on **CPU** after `fit`, so
the backend reclaims it (and the coreset) for the GPU.

**Where the pre-processing happens — settled 2026-08-26, and it was the unwanted answer.**
`AnomalibModule.forward` is `pre_processor → model → post_processor`, so the module pre-processes
whatever it is handed. `score()` had been pre-processing first, which resized and ImageNet-
normalised every image **twice**. It raised nothing, warned nothing, and still separated a defect
from a normal image — normalising twice is monotonic enough to preserve the ordering, which is
exactly why the smoke test and the separation stage both passed on the wrong path. Only the VisA
gate would have caught it, and nothing would have pointed here. `score()` now hands the model a raw
[0,1] tensor at native resolution and lets it pre-process once, as it does during `fit`.

The measurement that settled it is worth copying for the next backend: the inner `model.model` is
the raw network with no pre-processor attached, so `model(raw)` vs `model.model(transform(raw))`
separates "the outer forward pre-processes" from "it does not" by construction, with no tolerance
judgement. The first attempt at this test compared a raw and a pre-processed input and read
"the scores differ" as an answer — but `Normalize` is injective, so they differ in both worlds.
**A test that passes in both worlds is not evidence.**

**⚠️ The open risk for the VisA gate:** that pre-processor is `Resize([256,256]) + Normalize`, with
**no CenterCrop** and a fixed square resize rather than shorter-side-256. Classic PatchCore is
Resize(256) → CenterCrop(224). If the ±1pt gate misses, look there first — the playbook already
names preprocessing mismatch as suspect number one.

**How this was found, and the reason the probe exists.** Three of those fixes were made by inferring
from a traceback, and each revealed a new failure in the same constructor — the signal that guessing
had stopped working. What broke the loop was `notebooks/probe_patchcore.py`, a stage-by-stage harness
where each stage **writes down what it expects before running the smallest thing that tests it**, so
a surprising PASS is as visible as a FAIL. It also corrected a wrong conclusion — that the Lightning
Trainer was the wrong abstraction — by showing the candidates reached the coreset step and died in
the display layer. Use it, and extend it, for the next backend.

**One operational constraint, learned the hard way:** the author cannot paste from a terminal into
Colab. A fix is delivered by **pushing it to `master`**, where notebook cell 1.1 hard-resets and
picks it up; that is why `patchcore_backend.py` is a tracked `.py` and not a `%%writefile` cell.
Changing the notebook's cell structure instead costs a full reload (reinstall anomalib, re-download
the weights), so prefer changing tracked Python.

## PatchCore's gate is MVTec AD classic, not VisA (2026-09-16, protocol v0.2.12)

Found by reading both primary sources for the ±1pt gate, and it changed the acceptance criterion
for M2.

**PatchCore's own paper cannot report VisA.** arXiv:2106.08265 v2 is May 2022; the VisA dataset
(arXiv:2207.14315) is July 2022. The playbook's "the published-numbers table, from each method's
own paper" was unsatisfiable for this pairing.

**The only primary source for PatchCore on VisA is weak.** The VisA dataset paper's Table 6 gives
image AU-ROC 92.4 (1-class, averaged over 12 objects), but it is the only PatchCore result in that
paper — the per-object appendix tables are all PaDiM — it states none of its PatchCore
hyperparameters (no resolution, crop or coreset ratio), and its MVTec-AD control in the same row
is **99.8**, above every single-model number in PatchCore's own paper (99.0-99.1) and above its
99.6 ensemble. A miss against 92.4 would not distinguish a wrong implementation from a different
setup, and a criterion that cannot distinguish those is not a criterion.

**MVTec AD classic is the strong gate and was already in §2.** PatchCore's own paper, Table S1,
row `PatchCore-10` — and that suffix is the coreset subsampling ratio, so it is exactly this
repo's `coreset_sampling_ratio: 0.1`. **99.0 ± 1.0 I-AUROC, mean over the 15 categories.**

Protocol §2 v0.2.12 generalises the rule rather than special-casing PatchCore: the target must
come from the method's own paper at the configuration this repo runs; where only one dataset
satisfies that, the other is still run and reported with its discrepancy but cannot fail the
method. **Settle this per method, from its own paper, before its Colab run** — the other four have
not been read yet.

What was built with it, all CPU and all committed: `datasets/visa.py` and `datasets/mvtec_ad.py`
(both verified against the real archives), `datasets/registry.py` plus `run_eval.py --dataset`,
the pre-registered `configs/reproduction/patchcore_ref.yaml` (mean **and** the 15 per-category
values), and `scripts/reproduction_gate.py`. The gate refuses a category set that is not the
published one, a dataset or method mismatch, and pooled seeds; exit codes are **0 PASS, 1 FAIL,
2 refused**. It flags categories outside tolerance even when the mean passes — the case the
per-category targets exist for. Provenance:
`../vlm-anomaly-paper/docs/verified-literature-facts.md`, fourth pass.

## The gate FAILED by 0.057, and the decision is deferred (2026-09-16)

**Measured mean I-AUROC 97.94 against a published 99.0 — delta -1.057 against a ±1.0 tolerance.**
Full table: `results/reproduction/patchcore_mvtec_ad.md` (committed; a FAIL that lives only in a
closed session's scrollback is the same as no gate at all). Seed 0, Tesla T4, anomalib 2.6.0,
commit 647ca0b.

**One category decides the verdict.** `toothbrush` came in at 90.83 against a published 99.7, and
its -8.87 contributes **-0.591 of the -1.057** shortfall. Had toothbrush alone matched its
published figure the mean would be 98.53 — a comfortable PASS. The next four contributors are far
smaller: pill -0.141, cable -0.133, carpet -0.074, zipper -0.065. Ten of the fifteen categories
are inside ±1.0 and three are exact.

`toothbrush` is also the extreme of the set: **60 training images**, against 209 for the next
smallest, and a 42-image test split where one image moves I-AUROC by about 0.3 points.

### Two candidate causes, both pre-registered BEFORE this run

Neither is a post-hoc excuse, and the distinction matters — protocol §7 forbids tuning after
seeing results, so what may be changed is only what was already written down as suspect.

1. **Only one seed was run, and §6 requires three.** PatchCore is stochastic; that was measured
   on 2026-08-26, and the whole seed-provenance apparatus was built for it. A miss of 0.057 with
   the seed spread unmeasured is not yet a verdict — it is a result awaiting two more runs.
   **This is owed regardless of what else is decided**: no reported number may come from one seed.
   **This is now the only one of the two still standing** — see the refutation section below.
2. ~~**The CenterCrop deviation**~~ — **REFUTED 2026-09-17.** Recorded 2026-08-21 as "suspect
   number one if the ±1pt gate misses", quantified 2026-09-16 and tested 2026-09-17: running the
   classic transform gives **93.41**, i.e. it misses by 4.590 instead of 0.060. See the
   refutation section below.

### What is owed next, and one gap in the tooling

**Decided 2026-09-17: investigate, over all 15 categories.** The author chose the full grid
rather than the ~3-minute `toothbrush`-only probe: a 15-category run with the classic
pre-processor produces a gate score *directly comparable* to the failed one, where a single
category would only have said whether toothbrush moves, not whether the mean does. It costs the
session (~40 minutes of fit) on a hypothesis that is not yet confirmed; that trade was made with
the alternative on the table.

**Both runs get recorded, not just a passing one.** The phase-3 FAIL at
`results/reproduction/patchcore_mvtec_ad.md` is not overwritten; the classic run writes to
`patchcore_mvtec_ad_classic.md`.

**The tooling for it is built and on `master` (2026-09-17), so the next Colab session designs
nothing** — see "The classic pre-processor is now a switch" below.

**Tooling gap found by this run — CLOSED 2026-09-17.** `scripts/reproduction_gate.py` scored a
single seed and refused a root that pools seeds — correctly — leaving no way to score the
three-seed run §6 requires. It now takes `--seed N` (and `--seed unseeded`, which is not seed 0)
to score one seed out of such a root, and `--all-seeds` to score each separately and report the
**mean of the per-seed means ± their sample std**. The std is reported and never gates; §6 v0.2.13
settles that, because §6 asked for "mean ± std" without saying which number the verdict is on.
The required seed count is pre-registered as `n_seeds: 3` in the targets file, next to the
category count, and `--all-seeds` refuses a root holding any other number.

**Session state:** the Colab session was closed after 3.3. The shards were copied to
`MyDrive/reproduction/` so the gate can be re-scored, and a single category re-run, without
repeating the whole 40-minute fit.

## The CenterCrop risk is now measured, not hypothesised (2026-09-16)

The phase-3 run prints its coreset size per category, and those numbers settle the open
pre-processing question without a separate experiment.

Every category yields **exactly 102.4 coreset points per training image** — the implied train
counts come out as clean integers matching MVTec AD's published ones (bottle 209, cable 224,
capsule 219, carpet 280, grid 264, hazelnut 391, leather 245, metal_nut 220, pill 267, screw 320,
tile 230, toothbrush 60). Two things follow:

1. `coreset_sampling_ratio: 0.1` is genuinely applied, uniformly, across every category — not
   merely passed to a constructor that ignored it.
2. `102.4 / 0.1 = 1024` patches per image, i.e. a **32x32 feature grid**, which is what a
   **256x256 input** produces. Classic PatchCore's `Resize(256) -> CenterCrop(224)` would give
   28x28 = **784**.

So the adapter runs **31% more patches per image than classic PatchCore, over the full frame
rather than the centre crop**. This is the anomalib 2.6.0 pre-processor (`Resize([256,256]) +
Normalize`, no CenterCrop) recorded in the 2026-08-21 session note, now quantified. If the gate
misses, this is the measured difference to attribute it to; if it passes, it passes despite it.

## The classic pre-processor is now a switch, and the gate can score three seeds (2026-09-17)

CPU work, on `master`, 426 tests green. It exists so the next Colab session runs cells and reads
numbers instead of designing anything.

**`PatchCoreBackend(preprocess=...)`** takes `"anomalib"` (the default — 2.6.0's own
`Resize([256,256]) + Normalize`, a 32x32 grid, which is what every number measured so far came
from) or `"classic"` (PatchCore's `Resize(256) -> CenterCrop(224)`, a 28x28 grid). The name is
validated **before** the lazy anomalib import, so a typo in a notebook cell fails in a
millisecond instead of after the install, the weights and a 40-minute fit.

Three things about how it is built are deliberate:

- **It does not import a `PreProcessor` class from a documentation path.** anomalib 2.6.0's
  Engine API was verified on 2026-08-21; its PreProcessor API was not, and this backend has been
  bitten five separate times by a call that looked reasonable and had never been run. The classic
  branch reaches the PreProcessor anomalib itself built on the model and reuses that object's own
  `Normalize`, rather than re-stating ImageNet's constants from memory. If the shape it expects
  is not there, it raises naming what it actually found.
- **`preprocess` is declared by the backend and passed on by the adapter**, exactly like `seed`,
  and `PatchCoreRef` refuses a declared value its backend does not apply. This is the seed defect
  — a number in provenance that nothing applied — kept from walking back in through the other
  half of the configuration. It is NOT on the shared `BackendSeeded` contract: the
  256-vs-224 question is anomalib's PatchCore pre-processor specifically, and the other five
  adapters have no such choice to declare.
- **Probe stage 13 checks the consequence, not the API.** It fits 20 synthetic images under both
  pre-processings and compares coreset sizes, which must be 784 patches per image for `classic`
  against 1024 for `anomalib`. A transform assigned somewhere the forward pass never looks would
  pass an API check and still produce a complete, plausible, wrong run. Cell 3b.0 runs it and
  asserts before 3b.1 is allowed to start.

**Recorded now rather than discovered later:** a centre crop means the anomaly map covers only
the middle 224/256 of the frame, and `PatchCoreRef` then upsamples it across the **whole** native
image. The reproduction gate is image-level I-AUROC and is unaffected. **Any pixel-level number
from a `classic` run would be spatially wrong** — which is one more reason the choice is recorded
per shard.

**`scripts/reproduction_gate.py`** gained `--seed N` (and `--seed unseeded`, which is not seed 0)
and `--all-seeds`; see the tooling-gap paragraph above. It also now refuses a root whose shards
disagree about `preprocess`. That is not hypothetical: the shard filename carries only
dataset/method/category/seed, `run_id` keys map directories on `config_hash`, and the 2026-09-16
shards were copied into `MyDrive/reproduction/` to be re-scored — two pre-processings reach one
root the moment someone copies the wrong folder.

**One thing the approved design did not survive contact with:** `run_eval.py` was to gain a
`--preprocess` flag. It did not, because the CLI cannot run PatchCore at all — `build_method()`
constructs adapters by name and cannot inject a backend, so `patchcore_ref` from the CLI always
has `backend=None` and `prepare()` raises `MethodNotRunnable`. A flag there could never take
effect. The `preprocess` key goes into the `run_meta({...})` cfg in the notebook cell instead,
which is where phase 3 actually executes, and that is also what gives the classic run its own
`config_hash` and therefore its own map directory. (The usage example in the gate's docstring
showing `run_eval.py --method patchcore_ref` is aspirational for the same reason; it is not new,
and it is not fixed here.)

## Probe stage 13 passed (2026-09-17) — the PreProcessor API is now in the verified record

Run on a Colab T4, anomalib 2.6.0 under Python 3.13, seed 0. **This is the first time anomalib's
pre-processor API has been observed rather than inferred**, so all of it is recorded here.

**The untouched transform, printed verbatim:**

```
Compose(
    Resize(size=[256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
)
```

Two steps, no CenterCrop, and the resize is to a fixed `[256, 256]` — not shorter-side-256.
That confirms the 2026-08-21 session note from the object itself, and the ImageNet constants are
now attested rather than assumed.

**The API the backend reaches through is real:** `model.pre_processor` is a `PreProcessor`
(a Lightning callback — it carries `on_train_batch_start`, `setup`, `state_key` alongside the
`nn.Module` surface) and it exposes **`transform`**. The defensive introspection in
`_apply_classic_preprocessing` was the right call and cost nothing: it now runs against an API
that has been seen. Lightning lists `pre_processor` as child module 0 of the LightningModule
during `fit`, which is why one assignment reaches both `fit` and `score`.

**The measurement, which is the part that matters:**

| preprocess | coreset rows (20 images, ratio 0.1) | per image | grid |
|---|---|---|---|
| `anomalib` | 2048 | 102.4 | 32x32 |
| `classic`  | 1568 |  78.4 | 28x28 |

Both hit their predicted values exactly. The 102.4 reproduces the figure measured across all 15
categories on 2026-09-16, on 20 synthetic images instead — so the per-image patch count is a
property of the transform, not of the data, exactly as `preprocess_spec` assumes.

**Benign noise in that output, recorded so the next session does not chase it:** "Total length of
`DataLoader` across ranks is zero" (ValSplitMode/TestSplitMode NONE, expected), "Found 174
module(s) in eval mode at the start of training" and "`configure_optimizers` returned `None`"
(PatchCore does not train), and the unauthenticated HF Hub warning for the 276 MB timm backbone.

## Phase 3b.1 ran: all 15 categories under the classic pre-processor (2026-09-17)

Colab T4, anomalib 2.6.0, Python 3.13, seed 0, `preprocess=classic`, into
`results/reproduction/mvtec_ad_classic/`. Every category printed
`done (seed 0, preprocess classic)`.

**The coreset sizes confirm the transform reached all fifteen, not just the first.** The sampler's
progress bar totals `rows - 1` (established twice in stage 13: 2047→2048, 1567→1568), so the
coreset size is recoverable from the log, and every category lands on **78.4 points per training
image** — the classic 28x28 grid — against the 102.4 the anomalib transform gave on 2026-09-16:

| category | coreset rows | per image | anomalib would have given |
|---|---|---|---|
| bottle | 16385 | 78.4 | 21401 |
| cable | 17561 | 78.4 | 22937 |
| capsule | 17169 | 78.4 | 22425 |
| carpet | 21952 | 78.4 | 28672 |
| grid | 20697 | 78.4 | 27033 |
| hazelnut | 30654 | 78.4 | 40038 |
| leather | 19208 | 78.4 | 25088 |
| metal_nut | 17248 | 78.4 | 22528 |
| pill | 20932 | 78.4 | 27340 |
| screw | 25088 | 78.4 | 32768 |
| tile | 18032 | 78.4 | 23552 |
| toothbrush | 4704 | 78.4 | 6144 |
| transistor | 16699 | 78.4 | 21811 |
| wood | 19364 | 78.4 | 25292 |
| zipper | 18816 | 78.4 | 24576 |

284,509 coreset points in total, against 371,605 under the anomalib transform. This is a
**per-category** check, and it is the one that matters: a switch that silently applied to the
first backend only would have passed stage 13 and then produced fourteen categories of wrong
numbers.

**It also completes the train-count record.** The 2026-09-16 note derived twelve counts from the
coreset sizes and left three out; the same arithmetic gives **transistor 213, wood 247,
zipper 240**, which are MVTec AD's published figures. All fifteen are now accounted for.

**Cost:** the coreset fits ran 3s (toothbrush, 60 images) to 2m29s (hazelnut, 391), consistent
with the 30-60 minute budget the cell states.

## The CenterCrop hypothesis is REFUTED (2026-09-17, phase 3b.2)

**Pre-registered 2026-08-21 as "suspect number one if the ±1pt gate misses". Tested 2026-09-17.
The answer is no, and it is not close.**

| | anomalib (2026-09-16) | classic (2026-09-17) |
|---|---|---|
| mean I-AUROC | **97.94** | **93.41** |
| misses the ±1.0 gate by | 0.060 | 4.590 |

Report committed at `results/reproduction/patchcore_mvtec_ad_classic.md`, beside the phase-3 one.
Seed 0, Tesla T4, anomalib 2.6.0, commit `51c6404`.

**The failure is structured, not uniform.** Twelve of fifteen categories barely move (|delta| <
2.3). Three collapse:

| category | anomalib | classic | delta |
|---|---|---|---|
| capsule | 98.44 | **58.20** | **-40.24** |
| screw | 97.32 | **80.28** | **-17.04** |
| toothbrush | 90.83 | **84.44** | **-6.39** |

Those three are exactly the categories whose object is elongated or reaches the frame edge — the
capsule lies diagonally with its ends toward the corners, the screw is a thin diagonal whose tip
sits near the border, the toothbrush carries its bristle head at one end. `CenterCrop(224)` from
256 discards 12.5% of every edge, i.e. the region those categories are discriminated on. That is
a *hypothesis for the mechanism*, offered as such; the refutation itself does not depend on it.

**What this does and does not settle.**

- **Settled:** the CenterCrop deviation does not explain the 0.057 miss. Removing the deviation
  costs 4.5 points rather than recovering 0.06. The repo stays on the `anomalib` pre-processing,
  which is already its default — nothing is changed in response to a result, and nothing needs to
  be (protocol §7).
- **Also settled, and worth stating because it is counter-intuitive:** the 31% extra patches over
  the full frame are not a handicap here. They are why three categories work at all.
- **NOT settled:** our `classic` does not reproduce the paper's classic. The paper reports 99.0
  *with* a crop, including capsule 97.8 and screw 97.0, where ours gives 58.20 and 80.28. Either
  our classic is unfaithful in some way not yet found, or the paper's pipeline differs elsewhere
  such that the crop is harmless there. **This question is not opened.** Chasing it would mean
  tuning an implementation against a target after seeing its score, which is exactly what §7
  forbids; the pre-registered question was asked and answered, and the answer does not depend on
  which of those two readings is right.

**The one pre-registered cause left standing is the seed.** Only one was run and §6 requires
three. The anomalib configuration misses by **0.060** — small enough that the seed spread, which
has never been measured on real data, could plausibly cover it. That measurement is owed
regardless of what is decided about the gate: **no reported number may come from one seed**, not
even the number behind "PatchCore is flagged as not reproduced".

## The options for the next session, written down before it starts (2026-09-17)

The three-seed run is owed and is not one of the options — §6 forbids reporting any number from
one seed, including the number behind "PatchCore is flagged as not reproduced". What follows is
everything that *is* a choice, in the order it arrives.

### Decision 1 — CLOSED 2026-09-18: `SEEDS = (0, 1, 2)`

**All three seeds are re-run in the session; the 2026-09-16 seed-0 shards are not reused.** ~2 h
instead of ~80 min, bought for identical provenance across the three — same commit, same runtime,
same `preprocess` key on all three shards. The number that closes M2 gets it. Recorded in cell
3b.3's markdown as well, so the live session does not re-open a settled question.

The alternative, as it stood: `SEEDS = (1, 2)` reusing `MyDrive/reproduction/` (~80 min, seed 0's
provenance asymmetric to its siblings). It was numerically valid — the `anomalib` path is
untouched by the pre-processing work, because `preprocess_spec("anomalib")` returns
`center_crop: None` and never modifies the model, and the gate's `_check_single_preprocess`
ignores NaN so the older shards' missing `preprocess` column would not have tripped it. It was
declined on provenance, not on correctness.

### Decision 2 — after the three seeds: the gate verdict

Score with `--all-seeds`. The verdict is on the **mean of the per-seed means**; the std is
reported and never gates (§6 v0.2.13). Three outcomes:

**(a) The mean lands within ±1.0 → PASS.** M2 closes. PatchCore becomes the unflagged full-shot
anchor, phase 4 finishes (commit the notebook, confirm CI), and M3 — the full MVTec AD 2 grid —
unblocks. Note that a PASS reached this way is a PASS *despite* the pre-processing deviation, not
because it was fixed; §3.2 of the paper already says why that is worth stating.

**(b) The mean still misses → FAIL, and the choice is between two honest endings.**

- **Accept it.** PatchCore is flagged as not-reproduced in every table it appears in (§2). It
  still functions as the comparison anchor — the flag is a disclosure, not a disqualification —
  and WinCLIP starts immediately. This is the cheapest ending and it costs nothing in
  credibility, which is the whole reason §2 says "flag, don't drop".
- **Keep investigating**, under the constraint below.

**(c) The seeds disagree wildly** (a large std, some seeds passing and some failing). The verdict
is still on the mean, so this does not change the outcome — but it would be a finding in its own
right about PatchCore's stability at this coreset ratio, and it belongs in the paper's §5.5
whatever the verdict is.

### If the choice is "keep investigating": what §7 allows

**Both pre-registered causes are now spent.** CenterCrop is refuted; the seed will have been
measured. Anything further is a post-hoc hypothesis, and §7 does not forbid testing one — it
forbids *tuning* and it forbids reporting only the tests that helped. A new cause is legitimate
if it is **written down and dated before its test runs**, and if its outcome is recorded either
way. So the candidates below are pre-registered here, on **2026-09-17**, before any of them has
been run:

1. **The backbone weights may not be the paper's.** The 3b.1 log shows anomalib fetching
   `model.safetensors` (276 MB) from the HuggingFace Hub, which is timm's loading path, whereas
   the original PatchCore uses torchvision's ImageNet `wide_resnet50_2`. Those are different
   checkpoints from different training recipes, and PatchCore is entirely a function of its
   frozen features — so this is the largest remaining unexamined difference between our anchor
   and the paper's. **It needs a VERIFY step before it means anything**: print the resolved timm
   model id and checkpoint source in Colab, and compare against torchvision's. Do not act on the
   inference above; it is read off a download line, not off the library.
2. **The tolerance may be too tight for this particular mean.** `toothbrush` has 42 test images,
   where one image moves its I-AUROC by about 0.3 points, and it contributed more than half the
   original miss. A ±1.0 tolerance on a 15-category mean containing a category that granular may
   be measuring sampling noise. **This one may be reported but must not be acted on**: the
   tolerance was pre-registered, and widening it after seeing a miss is precisely what §7
   forbids. It is a limitation for §5.5, not a fix.

### VisA — deferred into the same session (decided 2026-09-17)

**Phase 3.4, the VisA secondary check, has still never been run**, and the author chose on
2026-09-17 to run it in the seeds session rather than on its own. It is now one typed line —
`!python scripts/run_visa_secondary.py`, a tracked script rather than notebook JSON, so a fix to
it reaches a live session through cell 1.1.

**It is not cheap.** An earlier note in this file called it "cheap relative to a 15-category fit";
that was wrong and is corrected here. VisA's one-class split carries **8,659 train normals across
12 objects** against MVTec AD classic's 3,629 across 15, plus 2,162 test images — roughly **two
hours**, comparable to the MVTec grid rather than a fraction of it.

**Order the session so a drop costs the least.** Both runs are resumable per category, but they
are not equally important:

1. **Setup** — phase 0, cell 1.1, cell 3.1 (~15 min).
2. **The three seeds on MVTec first.** They decide M2 and the gate verdict. `SEEDS = (0, 1, 2)`
   is ~2 h, `(1, 2)` with seed 0 restored from Drive is ~80 min.
3. **VisA last**, at seed 0 (~2 h). It is informational: it gives a second independent reading of
   the same implementation, which is worth having before choosing between the two endings under
   Decision 2, but it cannot change the verdict.

Total 3.5–4.5 hours, which is a long session for a free T4. Putting the seeds first means a drop
costs the reading, not the decision.

**One consequence to accept going in:** §2 v0.2.12 says VisA is *reported*, with its caveat. A
reported number falls under §6 like any other, so a complete VisA figure eventually needs three
seeds too — another ~4 h. Seed 0 alone is a first reading that tells you whether the
implementation is in the right region; it is not the number that goes in a table, and the script
takes `--seed` for the rest when that matters.

## Phase 2 ran green (2026-09-16) — plumbing only, nothing here is reportable

Vial end to end through `run_evaluation` on a Colab T4, anomalib 2.6.0 under **Python 3.13**
(the 2026-08-21 session did not record its Python version; 2.6.0 resolved again from the
`>=1.1` range three weeks later, so the unpinned range is stable for now — it is still pinned
at phase 4).

- **The sanity gate passes:** I-AUROC 0.947 on `regular`, 140 images over 7 lighting conditions
  at n=20 each, which is the whole of Vial's `test_public`.
- **The seed reached the sampler and the record:** Lightning printed `Seed set to 0` and the cell
  printed `declared seed: 0` — the first end-to-end exercise of the seed-provenance work on a real
  GPU method. Both paths carry the tag, checked by eye on the live session:
  `mvtec_ad2__patchcore_ref__vial__seed0.parquet` and a map directory
  `mvtec_ad2__patchcore_ref__12f01db684db__vial__seed0`, so a second seed can neither be mistaken
  for done nor overwrite the first's maps.
- **The float16 overflow guard did not fire**, so PatchCore's raw scores stay under 65504 on Vial
  and no §4 map-scale amendment is needed.
- **Measured cost, for M3 budgeting:** greedy coreset selection over Vial's 291 train images took
  **2m19s for 29797 indices** (~213 it/s). Scoring the 140 test images followed.
- **One warning, benign and annotated in the code:** torch warns that the PIL-backed array is not
  writable. `.float()` cannot alias a uint8 buffer, so it allocates and `.div_` mutates the copy.
  See the comment in `score()`; do not "fix" it with `arr.copy()`.

**The numbers are not a result.** Protocol §2: no MVTec AD 2 number is reported until PatchCore
reproduces its published VisA image-AUROC within ±1.0. They are also a single seed, one draw, of a
method this repo has measured to be stochastic. Recorded here only so a later run that disagrees
wildly is visible as a signal.

## PatchCore is stochastic, and the seed is now wired end to end (2026-08-26, closed 2026-09-07)

The probe's own numbers moved between two runs on byte-identical inputs — the same image scored
202.796432, then 203.890701. PatchCore's greedy coreset sampling is stochastic, and it is the first
method here that is: `intensity_baseline` is deterministic, so this never came up.

Two consequences, one closed and one open.

**Closed:** `PatchCoreBackend` now takes `seed=`, applied as
`lightning.seed_everything(seed, workers=True)` before `engine.train`. Probe stage 12 measures that
this actually reaches the sampler rather than trusting the name: two fits at seed 0 gave
69.406265 twice (delta 0.00e+00) and seed 1 gave 69.803741. **The spread between seeds is ~0.4 in
score units on synthetic data** — protocol §6's "three seeds, report mean ± std" is not a formality
here.

The default is `None`, deliberately not a silent `0`: see the open item for why a backend that
quietly seeds itself would make a false record look true.

**Closed 2026-09-07.** A method now declares the seed it applied (`AnomalyMethod.seed`), and
`run_evaluation` is the only writer of that field — `run_meta` no longer accepts one, so neither
entry point can record a seed nothing applied. Two further defects were found while fixing it and
are fixed with it: two seeds resolved to one shard path (so `is_done()` called the second one
finished and skipped it) and to one anomaly-map directory (`run_id` is `config_hash`, which carries
no seed, so the second overwrote the first's maps while the first's shard still pointed at them).
Both now carry a literal `seed<N>` / `unseeded` tag. The metric functions and
`scripts/calibrate_threshold.py` refuse a frame that pools seeds.

Still not built, deliberately: combining three seeds into mean ± std. That is reporting, it belongs
with `scripts/make_tables.py` (a `TODO(M5)` stub), and doing it wrong now raises rather than
publishing. Spec: `docs/superpowers/specs/2026-09-07-seed-provenance-design.md`.

## The threshold rule — built and pre-registered (2026-08-20)

The repo's one piece of unplanned work is done. Protocol v0.2.11 §4 pre-registers three
ground-truth-free rules on two axes (transform x calibration source), `alpha = 1e-3` with a
fixed sensitivity grid, and **`per_image_robust_z` as the designated private-split submission
rule** — fixed, with an a priori justification, before any result existed. `seg_f1_at` computes
the official SegF1 at a fixed threshold, pooled over a whole category as the official definition
requires; `seg_f1max` stays and stays marked **oracle**.

Calibration runs on the defect-free `validation` split via `scripts/calibrate_threshold.py`,
which refuses any other split, and writes `configs/thresholds/<dataset>__<method>.yaml`. **That
file is committed, and committing it before a submission is what makes the pre-registration
auditable.**

What still gates M4, in order:

1. **Submission packaging** — writing the server's payload (thresholded plus continuous maps in
   its format). Deliberately not built: the format is unverified until first login
   (`docs/datasets-access.md` leaves it as a checkbox), and guessing an upstream API is what
   protocol §3 forbids. Confirm the format on first login, then build it.
2. **Each method's VisA ±1pt gate**, which is the pre-existing ordering above and unchanged.

The paper's C1 and C3 `\todo{withdraw if the threshold rule is not built before submission}`
tripwires can come out once a first calibration has run on a real method — not on
`intensity_baseline`, which is the floor, not a detector.

## Companion paper — where it stands

`../vlm-anomaly-paper`, branch `master`. Abstract and **§1–§3 are written and reviewed**; §4–§6 are
stubs awaiting M3. Two things there that this repo's work must stay consistent with:

- `docs/verified-literature-facts.md` is the paper's provenance record — every literature claim
  traces to a primary source read directly, and it also records what was **not** verified. It
  corrected two assumptions that lived in *this* repo: "GroundingDINO + SAM" for SAA+ (now verified
  from the paper's §1/§2) and the belief that WinCLIP had been a VAND 3.0 entry (it was a Category 2
  baseline, on LOCO AD).
- It independently corroborates the SAA+ prompt-coverage contingency pre-registered in protocol
  v0.2.9: SAA+'s own paper reports on VisA, MVTec-AD, MTD and KSDD2 — **none of them MVTec AD 2** — so
  the realistic outcome is that no AD 2 category has a published prompt. Resolve
  `prompt_coverage: unresolved_until_first_colab_run` in `configs/methods/saa.yaml` at Colab phase A.3
  and decide then whether SAA+ stays in the study on those terms.

## External / parallel

- ~~**MVTec evaluation-server registration**~~ — **done. Access granted 2026-07-30**, seven days after
  the 2026-07-23 request and ahead of the 2026-08-06 follow-up deadline, so no follow-up was sent and
  the public-split-only scope reduction was never opened. The private split is in scope; the record
  and the remaining fields to fill on first login are in `docs/datasets-access.md`.
- **Dataset downloads for the grid:** only Vial (0.77 GB) is on disk. The rest of MVTec AD 2 is
  fetched per category on Colab when M3 runs (largest: Fabric, 10 GB); never into `/tmp` (tmpfs).
  The route is Drive, one category at a time, uploaded once each — notebook cell 22 and
  `docs/datasets-access.md`.

## Integration points every Colab playbook leaves to the executor

These cannot be pre-written without the data/repo in front of you, and each playbook marks them:
- ~~**The VisA loader**~~ — **built 2026-09-16** (`src/vlmab/datasets/visa.py`), against the real
  archive. `src/vlmab/datasets/mvtec_ad.py` was built with it, and both are in the dataset registry
  so `run_eval.py --dataset` reaches them.
- **The published-numbers table**, from each method's own paper — record the exact source next to
  it. **Done for PatchCore** (`configs/reproduction/patchcore_ref.yaml`). For the other four,
  read the paper first: PatchCore's case showed the table cannot be assumed to exist, and which
  dataset may gate a method is now a §2 v0.2.12 decision recorded before its Colab run.
- **The pinned versions/commits and checkpoint shas**, recorded back into the method's config and,
  for AnomalyCLIP, into the overlap audit's blank record-fields.

## Where to pick up (session handoff, 2026-09-17, after the phase 3b session)

**Phase 3b is done and its question is answered: the CenterCrop hypothesis is refuted** (section
above). Both gate reports are committed. Nothing about phase 3b needs re-running.

**The next move is cell 3b.3 — three seeds on the `anomalib` configuration**, and it is the last
pre-registered cause standing. The miss is **0.060**, the seed spread has never been measured on
real data, and §6 forbids reporting any of this on one seed. Read 3b.3's markdown before running:
it names the choice about where seed 0 comes from, which is the only thing left to decide.

Setup for that session, in order: phase 0 entire (fresh runtime — cells under "Phase 0", then
the restart-recovery cell if Colab asks), **1.1** (hard sync), **3.1** (extract from Drive), then
3b.3 and its scoring cell. Skip phase 1's other cells, phase 2, phase 3.2/3.3, and all of 3b.
Budget ~2 h for `SEEDS = (0, 1, 2)`; the runner is resumable, so a dropped session costs one
category.

**That skip list broke cell 3b.3, and it is fixed on `master` (2026-09-18, CPU).** 3b.3 had no
imports of its own — it inherited `run_evaluation`, `ResultStore`, `run_meta`, `PatchCoreRef`,
`PatchCoreBackend` and `dataset` from 3b.1, the cell the handoff tells you to skip. Following the
handoff exactly would have raised `NameError` on the first line of the loop, *after* the fifteen
minutes of install, Drive mount and extraction had been paid for. It now repeats the six imports
and builds its own `MVTecAD`, like every other run cell. `tests/test_notebook_run_cells.py` holds
the invariant: a cell that calls `run_evaluation` may inherit only `MVTEC_AD_CATEGORIES` (cell 3.1
defines it beside the extraction that no such session can skip) and must import everything else.
It parses the cells rather than matching strings, and it was checked against the pre-fix cell —
six orphan names — not only against the fixed one.


Then the gate decision comes back, informed rather than open. **The full decision tree — both
choices, all three outcomes, and the two post-hoc causes pre-registered on 2026-09-17 so that
testing them later stays legitimate — is written out in "The options for the next session" below.
Read that section before deciding anything.**

Both repos clean and in sync with `origin/master`; bench: 436 tests green on Python 3.11 and
3.13.

Older handoff (2026-08-26, after the second Colab session) follows.

**Phase 1 of the PatchCore playbook is closed.** The backend is built, verified stage by stage on a
Colab T4 (anomalib 2.6.0, torch 2.11.0+cu128), and reproducible under a fixed seed. Both repos are
clean and in sync with `origin/master`; bench: 313 tests green.

What that session actually found, in the order it found it:

1. The probe was testing a configuration we had already discarded — stages 8-10 still fed the model
   a PIL image and stage 7 never took the model back from the CPU. Both were fixed elsewhere on
   2026-08-21 and never back-ported. Found by reading, before the GPU ran.
2. anomalib's own `results/` tree tripped the sync cell's cleanliness assert mid-session. Fixed in
   `.gitignore` rather than by relaxing the assert.
3. **`score()` was normalising every image twice**, and had been since the backend was written. See
   the session note above. Nothing downstream would have caught it except the VisA gate.
4. **PatchCore is stochastic and `--seed` was recorded but never applied.** See the section above.
   Closed 2026-09-07: the backend takes a seed, the method declares it and the runner stamps it.

**The next move is phase 3**, and it is one Colab session with nothing left to design: upload the
15 MVTec AD classic `.tar.xz` to `MyDrive/mvtec_ad/`, then run notebook cells 3.1–3.4. Phase 2
closed green on 2026-09-16 (see its section above). A PASS closes M2 and unblocks phase 4, the
first real threshold calibration, and M3.

Fixes reach a live Colab session by being **pushed to `master`** — cell 1.1 hard-resets to
`origin/master` and prints what it synced. Push before asking for a re-run.

Two items only the author can close, both flagged in the paper's `references.bib` as `\todo` that
render as red text in the printed bibliography:
- the DOI for `duarte2026survey` (the author's own survey), and
- the full author list for `mllmzsad`.

One standing rule this project learned the hard way, worth restating: **a fact verified but not
recorded is a fact the next person cannot use.** Three times a claim was read from a primary source
and written straight into a task brief without landing in `verified-literature-facts.md`, and each
time the work correctly stalled until it was recorded. Record first, then write.
