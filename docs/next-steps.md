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
  methods, and lighting-grouped aggregation. Protocol frozen at **v0.2.15**.
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
   backend. **Its targets are read and pre-registered (2026-09-18) — see the section below.**
   Nothing about the acceptance criterion is left to the session: two gates, 91.8 on MVTec AD and
   78.1 on VisA, with per-category values for both. Watch: if it misses, the likely cause is
   anomalib's prompt ensemble differing from the paper (a §3 priority-2-source risk) — which is
   now a **count** (154 normal / 88 anomaly prompts) to check in phase A, not a suspicion to
   raise afterwards.
3. **AnomalyCLIP Colab phases (A–D)** — the Colab half of
   `docs/superpowers/plans/2026-07-25-anomalyclip-zeroshot-adapter.md`. Uses the **official repo**
   (anomalib does not ship it), so its backend is derived from the repo's `test.py` at a pinned
   commit. Use the **VisA-trained** checkpoint for MVTec AD 2 and the **MVTec-AD-trained** checkpoint
   for the VisA reproduction (overlap audit, §3.1). **Its targets are read and pre-registered
   (2026-09-18) — see the section below.** Two gates, one per checkpoint, and the paper's 518×518
   input into a 336px backbone is now in the method config as something a backend must apply.
4. **AdaCLIP Colab phases (A–D)** —
   `docs/superpowers/plans/2026-07-29-adaclip-zeroshot-adapter.md`. Zero-shot and, like AnomalyCLIP,
   **auxiliary-trained**, so it carries the same kind of overlap audit. Its CPU half is done and
   registered (`src/vlmab/methods/adaclip.py`, `docs/adaclip-overlap-audit.md`); what remains is the
   GPU backend. **Its targets are read and pre-registered (2026-09-19) — see the section below.**
   Two gates, 89.2 and 85.8, two checkpoints. ~~Watch: the plan pre-registers a contingency for
   the case where the repo publishes only one checkpoint — resolve it in Colab phase A.2~~
   **That contingency is RESOLVED, on CPU, from the repo's own README: both checkpoints exist.**
   What phase A.2 still owes is the sha256 of each file actually downloaded, and a refusal of the
   third checkpoint the repo ships, which trains on both MVTec AD and VisA.
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
6. **M3 — full MVTec AD 2 grid. ✅ COMPLETE for PatchCore, all 8 categories (2026-10-01).**
   Once each method's VisA gate passes, run the full public-test grid over all eight categories,
   one category at a time (the download/resume/shard unit). Mechanical. Vial and Sheet Metal are
   run and reported; **Sheet Metal put the anchor at chance**, which is a result, not a bug — see
   the handoffs below. **The anchor is now unusable on two of the four run** — Sheet Metal
   (at chance) and Wall Plugs (at or below chance in all six conditions) — and only one of them
   has an extreme aspect ratio, so that mechanism does not explain them. **The anchor is now
   unusable on three of the five run** — Sheet Metal, Wall Plugs and Can, the last of these
   *below* its own random baseline on `au_pro_005`. **Walnuts then worked, at the same
   2448×2048 as the failing Wall Plugs, which rules geometry out entirely.** The grid stands
   **3 usable anchors against 5 unusable.** The majority of MVTec AD 2 has no usable full-shot
   detection ceiling under the pinned configuration. See the M3 closeout below. The grid still
   has to be re-run per zero-shot method once each has a GPU backend.
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

### Decision 2 — RESOLVED 2026-09-18: outcome (a), the mean landed inside ±1.0

**98.02 ± 0.07, delta -0.98. PASS, by 0.02 points.** See "THE GATE PASSED" above for the numbers,
the caveats and what it does and does not close. The branches below are kept as the record of
what was on the table before the run — outcome (b)'s two honest endings were never reached.

### Decision 2 — after the three seeds: the gate verdict (as it stood)

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

## WinCLIP's gate is read and pre-registered — and it is TWO gates (2026-09-18, protocol v0.2.14)

CPU work, done while the PatchCore seeds session ran, because it does not depend on that verdict:
WinCLIP is the next method either way. **The paper was read directly** (arXiv:2303.14814v1 PDF,
21 pages) and recorded in `../vlm-anomaly-paper/docs/verified-literature-facts.md` (fifth pass)
**before** anything was written into a config — the ordering this project got wrong three times
before it stuck.

**The outcome is the opposite of PatchCore's.** PatchCore's paper predates VisA, so it had one
gate and one weak secondary check. WinCLIP's is CVPR 2023, postdates VisA, and reports both
benchmarks zero-shot at the backbone and window scales this repo runs. Under §2's rule — the
target must come from the method's own paper at the configuration this repo runs — **both
qualify, so both gate it**:

| | published (Table 1, `0-shot`, `WinCLIP (ours)`) | per-category source |
|---|---|---|
| MVTec AD classic (15) | **91.8** ± 1.0 I-AUROC | Table 10, column `K=0` |
| VisA (12) | **78.1** ± 1.0 I-AUROC | Table 16, column `K=0` |

All 27 per-category values are in `configs/reproduction/winclip.yaml`. Both lists were summed and
divided: 91.81 and 78.06, which round to the printed means — a transcription check, kept by
`test_winclip_per_category_numbers_average_to_their_published_means`.

**This confirms a number that had been carrying a warning label.** The paper repo's
"SEARCH-REPORTED BUT NOT PRIMARY-VERIFIED" list held WinCLIP's VisA 78.1 from a search summary.
It was right — and it stays struck through rather than deleted, because one correct summary is
not a reason to trust the next. AnomalyCLIP's 82.1 is still on that list, unread.

**Three things the paper settled that the plan carried as open risks:**

1. **The prompt ensemble is countable.** Figure 6 lists 7 normal state words, 4 anomaly state
   words and 22 templates, so the paper's ensemble is **154 normal and 88 anomaly prompts**.
   The plan named "anomalib's ensemble may differ from the paper" as the likely cause of a miss;
   it is now a count to check in phase A, before scoring, instead of a suspicion raised after.
2. **The pre-trained weights are a real open question.** The paper uses **LAION-400M** CLIP
   ViT-B/16+; what anomalib resolves to has never been observed. `configs/methods/winclip.yaml`
   now records both, the second as `unverified_until_first_colab_run`. This is WinCLIP's version
   of the backbone question still open for PatchCore — same architecture, different pre-training,
   entirely different frozen features.
3. **Table 7's VisA 78.9 is not the target.** It is the `+ specific states` ablation, which adds
   per-object defect words for PCB2/PCB4/Pipe fryum — the per-category prompt content §3 forbids
   us to write. It sits in the targets file under `not_the_target` precisely because it is higher
   than the real target and would be the tempting number to chase.

**Whether WinCLIP owes three seeds is a pre-registered measurement, not an assumption.** §6 says
three "where any stochasticity exists". The paper's 0-shot rows all carry ±0.0, but its five seeds
vary *shot sampling*, which zero-shot has none of — evidence about its pipeline, not ours. The
targets file pre-registers the procedure: run one category twice at two seeds; bit-identical image
scores mean one run is reportable with that measurement recorded beside it, anything else means
three seeds and `--all-seeds`. PatchCore is the standing reminder that a seed can be recorded
without being applied.

**The tooling gap this opened, and how it was closed.** `configs/reproduction/*.yaml` could
express one `gate` and one non-gating `secondary`, with the per-category map at the top level — a
shape that assumed one gate per method. `--which` now names a target block, and a block declares
`gates:` itself and may carry its own `per_category`/`n_categories`. **PatchCore's file is
untouched and loads exactly as before** (the legacy top-level keys are still read, and a test pins
that), which mattered because its three-seed run was in flight when this landed. Calling WinCLIP's
second dataset `secondary` would have been the cheap fix, and would have recorded it as the thing
§2 says cannot fail a method.

**Cost, for the session that runs it:** two full grids, not one — 15 MVTec AD categories plus 12
VisA objects, and VisA's one-class split is the larger of the two (8,659 train normals).


## AnomalyCLIP's gate is read too — two gates, two checkpoints (2026-09-18, protocol v0.2.15)

Same CPU session as WinCLIP's, same order: **arXiv:2310.18961v12 read directly, recorded in
`../vlm-anomaly-paper/docs/verified-literature-facts.md` (sixth pass) before anything reached a
config.**

| | published (Table 1, industrial, image-level AUROC) | per-category | checkpoint |
|---|---|---|---|
| MVTec AD classic (15) | **91.5** ± 1.0 | Table 12 | VisA-trained |
| VisA (12) | **82.1** ± 1.0 | Table 16 | MVTec-AD-trained |

Both lists sum to their printed means (91.49, 82.06). The 82.1 that sat on the paper repo's
unverified-leads list is confirmed — **both leads on that list turned out correct, which is not
evidence the shortcut works**, and the next paragraph is the counter-example.

**The checkpoint column is the new constraint, and it is not cosmetic.** AnomalyCLIP fine-tunes on
the *other* dataset's **test split** — "we fine-tune AnomalyCLIP using the test data on MVTec AD
and evaluate the ZSAD performance on other datasets. As for MVTec AD, we fine-tune AomalyCLIP on
the test data of VisA" (§4.1, repeated in A.1). So the two published numbers come from two
different models, and a run scored against the other one's number would fail a correct
implementation. Every block in `configs/reproduction/anomalyclip.yaml` names its checkpoint, a
test pins those against the overlap audit, and §2 v0.2.15 now requires it.

**The overlap audit gained the split it was missing.** It said "MVTec AD (classic)"; it now says
*test split*, in both rows. **No conclusion changed** — a test split of MVTec AD classic overlaps
VisA exactly as little as its train split does — but an auxiliary-training overlap audit that does
not say which split was trained on is not an audit, and a reader who checks the source will find
"we fine-tune ... using the test data" there. They should find it in our audit first.

**One bonus the pairing gives us:** the MVTec AD classic gate runs the **same checkpoint** as the
MVTec AD 2 primary evaluation (VisA-trained). That gate therefore exercises the exact configuration
the real results will come from, which is worth more than its ±1.0 verdict.

**Two configuration facts that were not in the method config and now are.** The paper feeds its
**336px-trained backbone 518×518 inputs** — deliberate, stated twice, "to obtain an appropriate
visual feature map resolution" — and applies a **Gaussian σ=4 smoothing to the anomaly map at test
time**. A backend that resized to 336, or that stripped the smoothing as "post-processing", would
be a different method from the one these targets were measured on. §4 forbids normalisation *we*
add, not the method's own published output.

**A gap recorded, not closed.** A shard can name its dataset, method, category, seed and
`preprocess`; it has **no field for a checkpoint**. Nothing mechanical would catch a run that
declared `visa_trained` and loaded the other one — the same shape as the seed that was recorded but
never applied, and the `preprocess` a shard could not name. The fix belongs with the AnomalyCLIP
backend when it is built: declare the checkpoint the way `PatchCoreBackend` declares its seed, and
have the adapter refuse a declared value its backend does not apply. Written into the audit's
first-run checklist.

**A worked example of why the primary-source rule is not pedantry.** AnomalyCLIP's Table 12 is a
convenient all-methods table that also prints WinCLIP's per-category MVTec AD numbers. Its
`toothbrush` cell says **88.0**; WinCLIP's own paper says **87.5**. Every other cell checked
matches and both columns still average 91.8, so nothing downstream moves. But WinCLIP's targets
were taken from WinCLIP's paper the day before, and had they been lifted from this table instead,
one pre-registered per-category target would now be wrong by 0.5 with nothing to reveal it.

## AdaCLIP's gate is read too — and the repo had to be read with the paper (2026-09-19, protocol v0.2.16)

Same CPU track as WinCLIP's and AnomalyCLIP's, same ordering: **arXiv:2407.15795v1 read directly,
recorded in `../vlm-anomaly-paper/docs/verified-literature-facts.md` (seventh pass) before
anything reached a config.** AdaCLIP is the fourth of five; only SAA+ is unread.

| | published (Table 1, industrial, image-level AUROC) | per-category | checkpoint |
|---|---|---|---|
| MVTec AD classic (15) | **89.2** ± 1.0 | Table 9 | `VisA & ColonDB` |
| VisA (12) | **85.8** ± 1.0 | Table 10 | `MVTec AD & ClinicDB` |

Both lists sum to their printed means (89.20 exactly, 85.76 → 85.8). Committed at
`configs/reproduction/adaclip.yaml` with ten tests holding the transcription and the invariants.

**This is the first method where the paper alone could not settle the gate.** AdaCLIP reports the
same two datasets *twice*: Table 1 under its own setup (89.2 / 85.8), and Appendix §4's Tables 6
and 7 "within the experimental setting of AnomalyCLIP" (89.6 / 83.9), which drops the medical
auxiliary dataset. §2's rule — the target comes from the method's own paper **at the configuration
this repo runs** — had to break a tie *inside one paper*, and the thing that breaks it is not in
the paper at all: **the official repo publishes weights for the main setting and none for the
ablation.** So the repo was read alongside the PDF, and the losing pair is recorded as
`not_the_target` (the MVTec one being the higher of the two, and therefore the tempting one).

**Three facts that come from the repo, not the paper, and all three matter.**

1. **The published-checkpoint contingency fired and resolved favourably.** The plan and the audit
   pre-registered "what if only one checkpoint is published?"; the README's weight table publishes
   three. Both of the checkpoints the audit assumed exist — but in its *third* branch, because
   neither is trained on exactly VisA or exactly MVTec AD classic: **each carries a medical
   auxiliary dataset too** (ColonDB, ClinicDB). The audit's rule for that branch says to record
   the complete training-set list rather than its nearest label and to set the caveat from the
   list alone; applied as written, `domain_proximity_caveat: false`. A config saying `visa_trained`
   was naming half of a checkpoint.
2. **The third published checkpoint must never be loaded.** `All Datasets Mentioned Above` trains
   on 14 datasets including **both** mvtec and visa — the demo weight behind the repo's
   HuggingFace Space. Loading it would put the test set's own family in the auxiliary data and
   void the zero-shot claim, and **nothing in a filename would reveal it**. It is now
   `forbidden_checkpoint: all_datasets` in both configs, with a test, and the backend must refuse
   it by name.
3. **The released checkpoints were selected by their score on the evaluation dataset.** Read from
   `train.py`, not from the prose: the val loader is built from `--testing_data` and `_best.pth`
   is saved on best pixel max-F1 there. So the checkpoint we run on MVTec AD 2 was *selected* on
   MVTec AD classic. No category is in the auxiliary **training** data — the audit's claim is
   untouched — but this is a second channel from the test set into the weights, it cannot be
   chosen around (both published checkpoints were made this way), and it now has its own row in
   the audit and its own line owed in the paper's §3.1 table.

**The one that has to be said before any number exists:** the repo states, verbatim, that its
released weights do **not** reproduce its published table — "the reported performance may vary
slightly compared to the detection performance with the provided pre-trained weights. Some
categories may show higher performance while others may show lower" — and that training was FP16,
unstable, and best-of-N on a validation set. Our gate scores *their weights* against *their table*.
**No other method here carries such a warning from its own maintainers.** It does not move the
target (§2 takes the number from the paper) and it is not an excuse; it is pre-registered as
`released_weights_caveat` with `report_with_verdict: true` precisely so that it is stated with a
PASS as readily as with a FAIL, rather than produced afterwards if the gate misses.

**One judgement recorded rather than made in a GPU session.** The audit's contingency says to set
the caveat true if any component of the training data "shares a provider or industrial-inspection
domain with the test set in use". Read literally that is satisfied by VisA, since VisA *is*
industrial inspection — which would flag the very pairing the audit's own table declares clean. A
rule cannot do both, so the operative reading is provider-and-family proximity. Written down in
the audit, with the ambiguity, because resolving a pre-registered sentence is not a thing to do
live at 2 a.m. on a T4.

**Cost, for the session that runs it:** two full grids like WinCLIP's — 15 MVTec AD categories
plus 12 VisA objects — and **two checkpoint downloads**, one per gate, with the third weight in
the same Drive folder that must not be the one loaded.

## SAA+ is read, and its gate DOES NOT EXIST (2026-09-19; decided 2026-09-20, v0.2.17)

The fifth and last paper. **arXiv:2305.10724v1 read directly, recorded in
`../vlm-anomaly-paper/docs/verified-literature-facts.md` (eighth pass, plus a same-day addendum)
before anything reached a config.** Every method's paper is now read.

**The finding is that SAA+'s own paper publishes no image-level number, on any dataset.** §5.1,
verbatim: ZSAS performance is evaluated on "(I) max-F1-pixel (Fp) ... (II) max-F1-region (Fr)".
No image AUROC, no image AP, no image F1. The task it addresses is zero-shot anomaly
*segmentation* and it is measured as such throughout.

What it does publish, Table 1 — the only performance table:

| metric | VisA | MVTec-AD | KSDD2 | MTD | Total |
|---|---|---|---|---|---|
| Fp (SAA+) | 27.07 | 39.40 | 59.19 | 35.40 | 34.85 |
| Fr (SAA+) | 14.46 | 49.67 | 39.34 | 30.27 | 34.07 |

Three properties of those numbers matter as much as their values:

1. **Both are computed "at the optimal threshold" — they are oracle metrics**, the class this
   repo marks `seg_f1max` with and which §4 already refuses to report as if achievable.
2. **There is no per-category breakdown anywhere.** Table 1 is per dataset and per defect type
   (texture/object); Table 2, the ablation, is texture/object/total. The per-category
   pre-registration WinCLIP, AnomalyCLIP and AdaCLIP all carry cannot be done here at any
   tolerance — there is nothing to transcribe.
3. **Fr is not computed by the released code.** In `utils/metrics.py`, `max_f1_region` is assigned
   `0` in the `else` branch, the call is commented out, and it is absent from the returned dict.
   The metric the paper proposes as its own contribution cannot be reproduced with the authors'
   own code as published.

**And the trap, named before anyone runs it:** the repo's `metric_cal` *does* compute image-level
AUROC (`i_roc`), from the max of the anomaly map. So executing their code produces an image AUROC
with no published counterpart. Treating it as a target would manufacture a criterion the
literature does not contain. `configs/methods/saa.yaml` records `image_score_from_map: max` as
*our construction following their code*, never as a reproduction.

### What this settles on its own, and is already committed

- **The paper's cited supplementary does not exist.** §5.1 points to it for the per-category
  prompts; v1 is the only version, 13 pages, ending at the references. That confirms **from the
  paper itself** the decision protocol §3 v0.2.9 already made — the repo is the only possible
  source for those prompts. It was the right call for a reason nobody had checked.
- **The branch is `SAA-plus`, not `master`.** `master` holds only the vanilla SAA demo, whose
  published VisA Fp is 12.76 against SAA+'s 27.07. A clone that does not name the branch runs a
  different method and says nothing about it. Now in both configs, with a test.
- **`prompt_coverage` is RESOLVED as `0_of_8`, on CPU** — the Colab phase A.3 question, answered
  without a session. The repo publishes parameters for MVTec AD classic, VisA, KSDD2 and MTD, and
  nothing else. For MVTec AD 2 this is an **absent dataset**, not a category that falls back, and
  `property_prompts` has no entry either — so §4.1.2's rules (object count, k_mask, area
  threshold) have no published values for our eight at all. That is the larger half of the gap:
  the language prompts fall back to three generic lines, the property rules fall back to nothing.
- **The property constraint is a POSITIONAL format, not prose.** `SAA/model.py` indexes
  whitespace tokens at [5] count, [6] similar/dissimilar, [7] object, [12] k_mask, [19] area
  threshold. Re-wrapping one, collapsing a double space or dropping its trailing space silently
  changes a threshold to whatever token lands at index 19 — no error, a complete and plausible run
  at parameters nobody chose. The 12 VisA entries were **generated from the repo source and
  round-tripped byte-for-byte rather than retyped**, and a test re-parses each at those five
  indices.
- **`visa_parameters.py`'s `official_prompts` is imported nowhere.** It is the larger and more
  official-looking of the two prompt tables in that file, and it is the dead one.

### DECIDED 2026-09-20 (protocol v0.2.17): no gate, and one is not invented

`configs/reproduction/saa.yaml` is written, and it contains **no gating block**. Two things take
the gate's place, because the gate's job splits in two for this method:

**(a) An integration-faithfulness check — the safeguard, and for SAA+ the stronger instrument.**
This repo does not reimplement SAA+: `anomaly_map_source: official_repo_final_map` stores the
repo's final map verbatim, so no arithmetic of ours can be wrong. The only failure mode left is
driving their code with the wrong configuration, and a published-number gate is an indirect,
noisy proxy for that. `scripts/saa_faithfulness.py` checks it directly — branch, pinned commit,
every VisA prompt byte-identical to the repo, the positional format still read at tokens
5/6/7/12/19, `hybrid_prompts.py` still using the live tables and not the dead `official_prompts`,
and the three-line fallback that all eight AD 2 categories depend on. Ten tests, most of them
breaking something on purpose. **It covers the static half only and says so on every run**; the
runtime half (400×400, K=5, N=400, the map stored unmodified, the fallback recorded when used) is
listed under `integration_faithfulness.runtime` and is owed by the backend.

**(b) Two NON-GATING checks against the Fp the paper does publish** — MVTec AD 39.40, VisA 27.07
— reported with their discrepancy and their confounds, unable to fail the method, on the
`gates: false` machinery §2 v0.2.12 built for PatchCore's VisA.

**The argument that earned (b) its cost is not the number.** MVTec AD classic is the **only**
configuration where SAA+ runs with its real per-object prompts (15/15); the primary MVTec AD 2
evaluation runs at 0/8 on the generic fallback. It is the single piece of evidence this study
will ever have that the cascade behaves as its authors describe.

**No tolerance is pre-registered for Fp**, deliberately. This repo has never measured that
metric's scale on this pipeline, and a number invented to fill the field is what §7 forbids. A
verdict there is read, not computed. Three confounds are written down before the run: the paper
measures at 400×400 against our native-resolution raw maps (v0.2.6); both sides are
oracle-thresholded, which makes the comparison apples-to-apples but means neither figure may be
reported as achievable (§4); and whether the paper's per-dataset figure is a mean over categories
or a pooled-pixel number is **not yet established** — settle it from the repo's eval loop before
reading any discrepancy. One thing does line up exactly: our `seg_f1max` and their
`calculate_max_f1` are the same construction — pooled pixels, sklearn `precision_recall_curve`,
maximum F1.

**§2 gained a general rule from this**, rather than a special case: where a method's own paper
provides no target in the gating metric, the method is neither dropped nor given a manufactured
gate. A gate's purpose is to detect a broken integration; where the literature cannot serve that
purpose, the integration is checked directly instead.

**Cost warning, with yesterday's VisA lesson fresh:** (b) is 27 categories through a
GroundingDINO+SAM cascade. The plan already pre-registers a measured cost probe (phase C.2) —
run it before committing a session, not after.

Whatever is decided, one thing is already true and belongs on every SAA+ table: on MVTec AD 2 it
runs **without its per-object prompts and without its property rules**, which is not SAA+ as
published. Protocol §3 v0.2.9 already requires that disclosure; the 0/8 makes it universal rather
than partial.

## M3 is costed, and the cost is NOT where it looked (2026-09-20)

Before any category was uploaded, the grid's cost was estimated — and the first estimate, made
from archive sizes, was wrong by an order of magnitude in the alarming direction. Fabric is 10 GB
against Vial's 0.77 GB, so extrapolating quadratically from Vial's 291 training images suggested
a single category could cost twenty hours. **The dataset paper's Table 4 refutes that**, and the
row for Vial was cross-checked against the archive on disk before the other seven were trusted
(provenance: paper repo, `verified-literature-facts.md`, 2026-09-20).

**Train sets are small and tightly clustered: 137 to 432 images.** The archive sizes are driven by
**resolution, not image count** — Fabric is 2448x2048 against Vial's 1400x1900, with *fewer*
training images. Full table in `docs/datasets-access.md`.

### The fit cost, from the two points we have measured

The exponent fitted across Vial (291 images, 2m19s) and VisA's candle (900, 23m39s), same GPU,
is **2.06** — quadratic, confirmed independently on two datasets. Extrapolating:

| category | train | fit/seed | x3 seeds |
|---|---|---|---|
| Sheet Metal | 137 | 0.5m | 1.5m |
| Fruit Jelly | 263 | 1.9m | 5.7m |
| Vial | 291 | 2.3m | 7.0m |
| Wallplugs | 293 | 2.3m | 7.0m |
| Rice | 313 | 2.7m | 8.0m |
| Fabric | 387 | 4.1m | 12.3m |
| Can | 412 | 4.6m | 13.9m |
| Walnuts | 432 | 5.1m | 15.3m |

**The whole M3 grid is ~1.2 h of coreset fitting.** Less than a third of what one seed of VisA
cost. Scoring, image I/O at these resolutions and map writing sit on top, and those *do* scale
with resolution — but the fit, which dominated everything so far, does not.

### Where the cost actually is

1. **Uploading 30.4 GB to Drive, one category at a time, by hand.** The archives sit behind
   mvtec.com's registration form, so no session can fetch them. **[SUPERSEDED 2026-10-01: the form
   issues a direct `mydrive.ch` link that needs no login or cookies, so a session CAN fetch them
   with `wget` — see `docs/datasets-access.md`. Everything below describes the cost as it stood
   before that was tried.]** This is the human bottleneck and
   the reason the runner takes one category and runs every seed before moving on: the alternative
   needs each category on Drive three separate times.
2. **Anomaly maps at native resolution.** Stored as float16 `.npy`, so a Fabric map is 9.6 MB
   and its 156 public-test images are 1.46 GB per seed. Across eight categories and three seeds
   the grid writes **24.9 GB of maps** — which does not fit Drive's free tier at all.
   `scripts/run_mvtec_ad2.py` therefore copies **shards only** to Drive and leaves the maps on the
   VM. That is safe because the image-level metrics are already in the shards and the maps are
   genuinely regenerable: same seed, same commit, same data, and the fit is now minutes rather
   than hours. Maps that a figure or the M4 submission actually needs get kept deliberately, per
   category, rather than by default.

   **The consequence, found on the first Vial run and now handled (2026-09-21):** the runner
   resumes on `ResultStore.is_done`, an existence check on a shard *on the VM* — so a fresh
   runtime re-fitted all three seeds while three good shards sat on Drive. `restore_from_drive`
   now copies back any shard Drive has and the VM does not, before deciding what to run, and
   never overwrites a local file with Drive's copy. A restored shard references maps that never
   left the VM that computed them, so `summarise` checks each `map_path` and nulls the missing
   ones — `pixel_metrics` already returns `{"n": 0}` for an empty column, so **the image-level
   table still comes out and only AU-PRO and SegF1 are skipped**, with a message saying which
   rows and how to get them back. Previously that case would have raised from inside `np.load`
   with no hint of why.

### The tool, and what it deliberately does not do

`scripts/run_mvtec_ad2.py --category <name>` fetches from Drive, verifies the layout with
`prepare_data.py`, runs seeds 0/1/2, copies shards to Drive **after every seed**, and prints a
per-lighting aggregation. Ten tests.

**It does not score the `validation` split.** `run_evaluation` re-fits per call, so a second split
roughly doubles the fit cost per seed; more importantly, the threshold rule's seed semantics —
one calibration per seed, or one for the seed that gets submitted — are **not pre-registered**, so
producing those scores now risks producing the wrong ones. The cost of that decision is a second
upload of each category when M4 runs. Recorded rather than absorbed, and pinned by a test so it
is not quietly added.

### ⚠️ The pre-processing note that now applies to every AD 2 number

**Not one of the eight categories is square, and Sheet Metal is 4224x1056 — a 4:1 frame.**
anomalib 2.6.0's pre-processor is a fixed square `Resize([256, 256])` with no aspect-ratio
preservation, read off the transform object itself on 2026-09-17. MVTec AD classic's images are
square, so **no reproduction number in this repo is affected; every MVTec AD 2 number will be**,
and Sheet Metal most extremely.

This is a property of the pinned configuration the gate was passed at, not a defect to fix after
seeing a result (§7). It is recorded **before the first AD 2 number exists** so that it is a
stated property of the study rather than an explanation produced afterwards, and it belongs in the
paper's Threats to Validity — where, unlike the field-level findings in §5.E, it genuinely is one.
It is also the same mechanism registered on 2026-09-19 as an untested hypothesis for the VisA
shortfall; VisA's released image dimensions are still unverified, MVTec AD 2's are now
primary-source confirmed.

## VisA ran: 86.26 against a published 92.4 (2026-09-20) — reported, cannot fail the method

Twelve objects, none dropped, seed 0, Tesla T4, anomalib 2.6.0, `preprocess=anomalib`, commit
`8f043c7`. Report committed at `results/reproduction/patchcore_visa.md`. ~4 h, as measured the day
before. **Delta -6.14.**

**What this is not: a verdict.** PatchCore's gate is MVTec AD classic and it PASSED (98.02 ± 0.07,
2026-09-18). §2 v0.2.12 makes VisA a reported secondary check that cannot fail the method, and the
reason was pre-registered long before this number existed: the 92.4 comes from the VisA dataset
paper's Table 6, a third-party source that states none of its PatchCore hyper-parameters, and
whose MVTec-AD control in the same row is **99.8** — above every single-model number in
PatchCore's own paper. A miss against it cannot distinguish a wrong implementation from a
different setup. That is why it does not gate, and nothing about the decision changes now that the
number is unflattering.

**The train-count record closes EXACTLY.** The last four objects give pcb2 901, pcb3 905, pcb4 904,
pipe_fryum 450, and the twelve sum to **8659** — VisA's documented 1-cls total, to the image. Every
object's coreset landed on `floor(N x 102.4)`, so the `anomalib` 32x32 grid reached all twelve.
Whatever -6.14 means, it is not "the transform failed to apply", and that was worth establishing
before anything else.

**Is it evidence our implementation is wrong? The check cannot say, but the surrounding evidence
leans away from it.** The same code, the same pre-processing and the same seed path reproduced
MVTec AD classic within 1 point across three seeds. A broken PatchCore that is 1 point off on one
dataset and 6 points off on another would be an odd kind of broken. That is an argument, not a
proof, and the next paragraph is the class of defect it would not cover.

### A post-hoc hypothesis, recorded 2026-09-20 BEFORE any test, and NOT acted on

Protocol §7 forbids tuning after seeing a result and forbids reporting only the tests that helped.
It does not forbid writing down a candidate cause, dated, before testing it. This one is
pre-registered here on the day the result arrived.

**anomalib 2.6.0's pre-processor is `Resize([256, 256])` — a *fixed square* resize, with no
aspect-ratio preservation and no CenterCrop.** That was read off the transform object itself on
2026-09-17 and is in the verified record. MVTec AD classic's images are square, so a square resize
is harmless there and no MVTec run could ever have surfaced this. **If VisA's released images are
not square, the same transform distorts every VisA image**, and the distortion would sit in every
VisA number while sitting in no MVTec number — which is the shape of the discrepancy.

**It is NOT verified, and must not be repeated as though it were.** The VisA paper
(arXiv:2207.14315) states only the acquisition sensor — "All images were acquired using a
4,000 x 6,000 high-resolution RGB sensor" — and says nothing about the dimensions of the released
images. This repo does not record them either; `docs/datasets-access.md` has VisA's archive size
and layout but no image geometry, where it does record MVTec AD 2's.

**Settling it is one line against the data**, and it belongs in the next session that has VisA on
disk:

```python
from PIL import Image; import glob
print({p.split('/')[-4]: Image.open(p).size
       for p in sorted(glob.glob('data/visa/*/Data/Images/Normal/0000.JPG'))})
```

Record the answer in `docs/datasets-access.md` first, as every other geometry fact was. **Even if
it confirms non-square images, nothing is reconfigured in response**: that would be tuning against
a result, and this result does not gate. What it would license is a *dated, separately recorded*
experiment whose outcome is reported either way — the same footing the CenterCrop hypothesis had,
which was tested and REFUTED.

**One benign coincidence, named so nobody chases it:** `fryum` scores 86.26 and the mean over the
twelve is also 86.26. The other eleven average 86.2609 on their own, so this is arithmetic luck,
not a mean leaking into a category cell.

**Spread:** 29.4 points, from `macaroni2` 68.83 to `chewinggum` 98.24. Also unremarked-on by the
target, which publishes no per-category breakdown at all.

**Still owed if this number is ever reported anywhere:** §6 requires three seeds, and this is one.
At the measured ~4 h per seed that is ~8 h more, for a check that cannot change a verdict — the
budget question already flagged on 2026-09-19, now with a concrete number attached to it.

## THE GATE PASSED (2026-09-18) — PatchCore reproduces, by 0.02 points

Three seeds, `anomalib` pre-processing, Tesla T4, anomalib 2.6.0, commit `45462ea`. Report
committed at `results/reproduction/patchcore_mvtec_ad_3seed.md`, beside the phase-3 FAIL and the
classic run, none of which are overwritten.

**Measured mean of the per-seed means: 98.02 ± 0.07 against a published 99.0 — delta -0.98
against a ±1.0 tolerance. PASS.**

| seed | mean I-AUROC |
|---|---|
| 0 | 97.94 |
| 1 | 98.08 |
| 2 | 98.02 |

**The verdict rule was fixed before the run** (§6 v0.2.13, written 2026-09-17): the centre is the
mean of the per-seed means, the std is reported and never gates. So this is a legitimate PASS
under a pre-registered criterion. **It is also a PASS by 0.02 points, and every place this number
appears has to say so.** A reader who is told "PatchCore reproduced" and not told the margin has
been told something true and misleading.

**Seed 0 came back at 97.94, exactly the 2026-09-16 figure.** That was the free consistency check
on the `preprocess` work: `preprocess_spec("anomalib")` returns `center_crop: None` and never
modifies the model, so the path had to be bit-identical, and it was. Had it moved, that would
have mattered more than the verdict.

**What the seed actually did, stated precisely.** The last pre-registered cause is now measured:
the spread across three seeds is **0.07**, which is small. The pass did not come from a wide seed
distribution rescuing a miss — it came from the mean of the three (98.02) sitting **0.08 above
seed 0 alone** (97.94), which was enough to move -1.057 to -0.98. That is the pre-registered rule
doing exactly what it says, on a spread far narrower than the 0.4-in-score-units the synthetic
probe suggested in 2026-08-26. **This is the first measurement of PatchCore's seed spread on real
data**, and it belongs in the paper's §5.5 whatever else is said.

**It is not an even reproduction, and that is the finding worth more than the verdict.** Six of
fifteen categories sit outside ±1.0 on their own:

| category | measured | published | delta |
|---|---|---|---|
| toothbrush | 91.48 ± 0.58 | 99.7 | **-8.22** |
| pill | 94.17 ± 0.30 | 96.0 | -1.83 |
| cable | 97.85 ± 0.42 | 99.4 | -1.55 |
| zipper | 98.38 ± 0.13 | 99.5 | -1.12 |
| tile | 100.00 ± 0.00 | 98.9 | +1.10 |
| carpet | 97.62 ± 0.16 | 98.7 | -1.08 |

**`toothbrush` alone contributes -0.548 of the -0.98 shortfall.** It has been the story since
2026-09-16 and three seeds did not move it: 60 training images, a 42-image test split where one
image is worth ~0.3 points, and now a measured seed std of 0.58 — the largest in the set, on the
category that decides the margin. Without its shortfall the mean would be ~98.57.

**What this closes and what it does not.**

- **Closed:** PatchCore's reproduction gate (protocol §2). It is the unflagged full-shot anchor.
  Both pre-registered causes of the original miss are now spent: CenterCrop refuted 2026-09-17,
  the seed measured 2026-09-18. Neither explained it; the tolerance did.
- **Closed:** the tooling question. `--all-seeds` was used in anger and produced the verdict, the
  spread and the per-category spreads in one report.
- **NOT closed:** why our reproduction is uneven, and `toothbrush` in particular. Protocol §7
  does not forbid investigating it — it forbids tuning after seeing results, and it forbids
  reporting only the tests that helped. The two post-hoc causes pre-registered on 2026-09-17 (the
  timm-vs-torchvision backbone, and the tolerance's granularity on a 42-image split) are still
  written down, dated and untested. Testing either remains legitimate; neither may change a
  configuration.
- **Still owed for M2 as the README defines it:** the README's M2 line says "the GPU backends and
  the ±1pt VisA reproduction" — i.e. *every* method — while item 1 of this file's ordered list
  says PatchCore's gate closes M2. **Those two are not the same claim**, and the milestone
  checkbox is deliberately left unticked until the author settles which one M2 means. What is
  unambiguous: PatchCore's gate passed, and M3 is unblocked for PatchCore.

**Next in the same session: VisA (cell 3.4).** Still never run, still ~2 h, informational —
§2 v0.2.12 makes it a reported secondary check that cannot fail the method, and its 92.4 target
comes from a third-party paper that states none of its PatchCore hyperparameters. It is now a
second independent reading of an implementation that has passed its real gate, rather than
evidence for a decision.

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
  it. **Done for PatchCore** (`configs/reproduction/patchcore_ref.yaml`) **and for WinCLIP**
  (`configs/reproduction/winclip.yaml`) **and for AnomalyCLIP**
  (`configs/reproduction/anomalyclip.yaml`) **and for AdaCLIP**
  (`configs/reproduction/adaclip.yaml`). For the remaining one — SAA+ — read the paper first:
  PatchCore's case showed the table cannot be assumed to exist, WinCLIP's showed it can be richer
  than expected, AnomalyCLIP's showed a method can need **a different checkpoint per gate**,
  AdaCLIP's showed a paper can print the same number twice under two setups and need its own repo
  read to break the tie, and which dataset may gate a method is a §2 v0.2.12 decision recorded
  before its Colab run. **SAA+ closes the list and closes it oddly: its paper has no image-level
  table to take a target from at all** (2026-09-19), so what is recorded for it is the absence
  plus the two oracle segmentation metrics it does publish.
- **The pinned versions/commits and checkpoint shas**, recorded back into the method's config and,
  for AnomalyCLIP, into the overlap audit's blank record-fields.

## Session log — 2026-09-18

One index of a long day, because the detail lives in five separate sections below and a reader
resuming cold should not have to find them. Commits are on `master` in both repos.

**GPU (the author's Colab session), in order:**

1. **Cell 3b.3 — three seeds on the `anomalib` pre-processing.** 45 runs (3 seeds × 15
   categories), none dropped. Coreset sizes confirmed the `anomalib` transform reached every
   category (bottle 21400 → 21401 rows → 209 × 102.4, the 32×32 grid).
2. **The reproduction gate PASSED**: 98.02 ± 0.07 against a published 99.0, inside ±1.0 by
   **0.02 points**. Report committed (`results/reproduction/patchcore_mvtec_ad_3seed.md`, commit
   `cf00207`). Full detail, caveats and what it does *not* close: **"THE GATE PASSED"** below.
3. **VisA was not started.** The seeds took the session; VisA is ~2 h on its own.

**CPU (this side), in order:**

1. **Fixed cell 3b.3 before it ran** (`5e43da8`). It had no imports of its own and would have
   raised `NameError` on the first line of the loop, after the ~15 minutes of setup were already
   paid for. `tests/test_notebook_run_cells.py` now parses the notebook and holds the invariant.
2. **Closed Decision 1** (`45462ea`): `SEEDS = (0, 1, 2)`, all three re-run for identical
   provenance rather than restoring seed 0 from Drive.
3. **Read WinCLIP's paper and pre-registered its gate** (`96c0446`, paper repo `bdf492b`).
   **Two gates**, 91.8 and 78.1, per-category for both. Protocol → **v0.2.14**.
4. **Read AnomalyCLIP's paper and pre-registered its gate** (`e820700`, paper repo `08355af`).
   **Two gates, two checkpoints.** Protocol → **v0.2.15**. The overlap audit gained the *test
   split* it was missing.
5. **The targets schema learned to hold more than one gate** — `--which` names a block, a block
   declares `gates:` itself. PatchCore's file is untouched and its legacy keys still load, pinned
   by a test, because its three-seed run was in flight at the time.

**State at close:** both repos clean and in sync with `origin/master`; bench 452 tests green on
3.11 and 3.13; protocol v0.2.15; three-seed shards saved to `MyDrive/reproduction_3seed/`.

**The one thing that must not be lost in the retelling:** the gate passed *by 0.02 points*, on a
reproduction that is **not even** — six of fifteen categories sit outside ±1.0 and `toothbrush`
(-8.22) contributes -0.548 of the -0.98 by itself. Every table that cites PatchCore as the anchor
owes both facts.

## Fruit Jelly ran, and it settled three things besides the category (2026-09-29)

The category's own result is in `results/mvtec_ad2/patchcore_ref/fruit_jelly.md`: detection swings with the
lighting (I-AUROC 0.684–0.853) while localisation does not move (AU-PRO@30% 0.6295–0.6453,
per-seed ± of 0.001–0.002). That is **not** the Vial pattern, where both metrics fell together
under `underexposed`, so localisation-under-lighting is a per-category question on the evidence of
two, not a finding.

### 1. The seeding reproduces across sessions and machines — first evidence

The category had already been run on **2026-09-20 at commit `a7bc7bd`**, in the same session as
Sheet Metal, and no report was ever written. The 2026-09-29 session restored those shards from
Drive, skipped every seed, and re-derived the same four numbers. They were then deleted and the
category re-run from scratch on a different VM, nine days later. **All four I-AUROC values came
back identical to four decimals** (0.8533 / 0.7911 / 0.6844 / 0.8044).

`git diff a7bc7bd..61704b4 -- src scripts configs` is empty, so this tests the seed, not the code.
It is the first time `PatchCoreBackend(seed=n)` has been shown to reproduce end to end across
sessions and machines — the seed-provenance work of 2026-09-07 wired it, and this is the check
that it holds. Both runs used a Tesla T4; nothing here says anything across GPU models.

### 2. The coreset formula in the reports is off by one, three times out of three

`floor(N × 102.4)` predicts 26931 for Fruit Jelly's 263 training images. The fit reported
**26930** — and Vial (29797 against 29798) and Sheet Metal (14027 against 14028) are the same one
step low. The prediction was written down before this run, so it is a confirmed pattern rather
than a noticed coincidence.

**What is NOT established:** whether the memory bank itself holds one fewer patch, or whether
anomalib's `Selecting Coreset Indices` progress bar counts one iteration short of what it selects.
All three figures were read off that bar. One line settles it — `model.memory_bank.shape[0]` after
a fit — and it costs nothing on top of a category that is fitting anyway. Do it on `wallplugs`,
and correct the three reports once the answer is known rather than now.

**ANSWERED 2026-09-30, and the conclusion is the opposite of what the section above implies.** A
fit on 30 synthetic 256×256 images: the bar ran **3071** iterations and `_model.model.memory_bank`
came out **(3072, 1536)**, against `int(0.1 × 30 × 1024) = 3072`. So **the memory bank is exactly
`floor(N × 102.4)` and the progress bar runs one short of it.**

The formula in the reports was right all along; **the three figures were wrong**, because all three
were read off the bar and recorded as bank sizes. Corrected in place, with the reason, in
`vial.md` (29797 → 29798), `sheet_metal.md` (14027 → 14028) and `fruit_jelly.md` (26930 → 26931).
Wall Plugs is 30003 by formula and was never observed.

The 1536 is a free cross-check: `layer2` + `layer3` of `wide_resnet50_2` is 512 + 1024, so the
bank's second dimension is the concatenated feature width the config asks for.

**A plausible mechanism, NOT verified:** a greedy k-center sampler picks its first centre before
the loop and iterates for the remaining `k − 1`, which would produce exactly this. Nothing here
read anomalib's sampler, so that stays a guess — the measured relationship is what to rely on.

**Deliberately not recorded in `src/`.** A comment in `patchcore_backend.py` would be the obvious
home, and it would also make every remaining category run at a different code commit than the four
already run. The 2026-09-29 determinism check depended on there being *no* code difference across
a span of commits; that property is worth more than a comment's convenience. `src/` stays untouched
until M3 is finished.

### 3. A defect in `run_mvtec_ad2.py`, found by using it

`write_report` stamps `git rev-parse HEAD` of the machine writing the report. On a resumed run
that is **the wrong commit**: the 2026-09-29 session wrote a Fruit Jelly report claiming `61704b4`
for numbers computed at `a7bc7bd`. The shards carry the truth — `run_meta` stamps a `commit`
column on every row — so the fix is to read it from there, and to refuse to write a single-commit
provenance line when the shards disagree among themselves. Owed, not yet done. The report now in
the repo is from the clean re-run, so its commit line is correct.

**The wider gap it exposes:** `tests/test_results_are_tracked.py` guards a report from being
silently gitignored, but nothing guards against a run that produced shards and *no report at all*.
That is how Fruit Jelly's first run vanished for nine days. The check would have to live where the
shards are, not in the repo, so it is not obviously a test — recorded as a known hole.

## Wall Plugs: the anchor fails again, and this time aspect ratio cannot be the reason (2026-09-29)

Full result in `results/mvtec_ad2/patchcore_ref/wallplugs.md`. **All six lighting conditions are at or below
0.50 I-AUROC** (mean 0.447, best 0.5000, worst 0.3956) while **AU-PRO@30% runs 0.2773–0.5132
against a random baseline of 0.1487**. The maps find the defects; the image-level score does not
rank the images containing them. Fruit Jelly showed this in one condition; here it is the category.

**Two of the eight categories now have an unusable anchor, and the pre-registered mechanism covers
only one.** Wall Plugs is 2448×2048, 1.20:1 — the mildest reshaping in the dataset after Vial,
where the anchor works well. Sheet Metal's 4:1 compression has nothing to say about this failure,
so "the square resize breaks the anchor" is not a general explanation and must not be written as
one. It also is not a lighting story: `regular` itself is 0.4600.

### AU-PRO now has a reference point, and it is measured

`au_pro_005 = 0.0292` on `shift_2` was unreadable — good or terrible depends on what random looks
like, and nothing in this repo said. It is derivable (for an uninformative map PRO(fpr) ~ fpr, so
the normalised area over [0, L] is L/2) and now it is **measured through this repo's own `au_pro`**:
**0.1487 at the 30% limit, 0.0249 at 5%**, five trials of fifteen images.
`tests/test_au_pro_random_baseline.py` pins it, so a change to normalisation, thresholding or
connectivity fails a test instead of quietly moving the reference every published AU-PRO number is
read against. `shift_2`'s 0.0292 is within 17% of uninformative.

### The "~20 images per condition" rule is dead, and it cost a failed summary

It held three times — Vial 20, Sheet Metal 19, Fruit Jelly 20 — and Wall Plugs has **25**. Both
numbers vary: conditions go 7 → 6 → 4 → 6, images per condition 20 → 19 → 20 → 25. The 8.02 GB
projected for 2448×2048 was computed at 20 images; the real peak was **10.03 GB**, so the guard
tripped above the budget the projection implied. Corrected in `docs/datasets-access.md`, with the
remaining 2448×2048 categories now budgeted at 25 until their counts are printed.

### A second defect in `run_mvtec_ad2.py`, in the recovery path

When the guard trips, the script prints a suggested command. On Wall Plugs that suggestion was
wrong twice over:

* **it omits `--summarise-only`**, so the retry re-enters the seed loop and holds ~3 GB of torch
  while aggregating — the exact overhead the flag exists to shed, documented in this file since
  2026-09-20;
* **its budget comes from available RAM, not from the requirement it just computed.** It suggested
  9 GB for a peak it had measured at 10.03 GB, so following it verbatim fails again.

The fix is mechanical: suggest the computed requirement with headroom, include the flag, and say
plainly when the requirement exceeds what the machine reports. The run went through with
`--summarise-only --max-bytes 11000000000`. Owed, with the `write_report` commit-stamp defect from
the same week.

## Can: the anchor scores below random, and two mechanical causes are refuted (2026-09-30)

Full result in `results/mvtec_ad2/patchcore_ref/can.md`. **All six conditions well below chance on I-AUROC**
(0.326–0.443, mean 0.388, with the training condition `regular` the *worst* of the six), and
**`au_pro_005` averaging 0.0024 against the measured random baseline of 0.0249** — a tenth of
uninformative, with three conditions at exactly 0.0000. `au_pro_030` is the only metric above its
baseline (0.298 against 0.1487).

**A number below its own random baseline is a claim about anti-correlation, not difficulty**, so it
was not written down until two mechanical explanations had been tested. Both were run in the same
session, with the maps still on the VM, and both are recorded either way:

* **map/mask geometry — REFUTED.** Both are `(1024, 2232)`; same shape, same orientation, no
  transpose. Eight anomalous images at seed 0.
* **the map lighting up the frame border — REFUTED.** The fraction of each map's top 1% of values
  in the outer 5% band is **0.000 across all twelve images sampled**, six normal and six anomalous,
  where a diffuse top 1% would give ~0.19. The hottest regions are entirely interior. That is the
  healthy behaviour — the background is the most consistent region across training images, so the
  least anomalous.

What the two establish together: the map responds to the object and not to its defects, and on five
of eight sampled anomalous images the mean map value *inside* the ground-truth region is lower than
outside (0.796–0.880 against 1.047–1.404 for the other three). The anti-correlation is in the
scores, not the geometry. **Why is not established and no hypothesis is offered** — testing one now
would be searching for a configuration that improves a number after seeing it (§7). A
pre-registered mechanism as a separate dated experiment is the only route open, and none was
registered for this category.

### The "direction shifts cost localisation" pattern has a second reading

On both Can and Wall Plugs the three `shift_*` conditions score below the three intensity
conditions on `au_pro_005` — Wall Plugs 0.0985/0.0292/0.0753 against 0.1688/0.1723/0.1694, Can
0.0001/0.0000/0.0000 against 0.0036/0.0028/0.0077 — and on `au_pro_030` here as well (shifts 0.261,
intensity 0.336). Two categories in the same direction, both of them ones where detection has
failed, so the reading rests on localisation alone. Vial degraded on both metrics; Fruit Jelly on
neither. Still not a finding; now worth watching deliberately.

### A fourth defect, and this one has cost three categories

**The coreset size is provenance and it is being read off a tqdm bar.** It was missed on Wall
Plugs, Can and (as a bank figure) on all three before them, and reading the bar instead of the bank
is what put a wrong number in three reports. `PatchCoreBackend` knows
`self._model.model.memory_bank.shape[0]` the moment `fit` returns; `PatchCoreRef` already stamps
`seed` and `preprocess` into every shard's provenance and should stamp this the same way. Then no
human copies anything and the figure is in the parquet next to the numbers it describes.

Owed, with the `write_report` commit-stamp defect and the guard's wrong recovery suggestion. **All
three are deliberately not being fixed until M3 finishes**, for the reason in the 2026-09-30
coreset section: the four categories already run share a code tree with no differences across
their commits, and that property is what made the determinism check possible. `src/` stays frozen.

## Walnuts works, at Wall Plugs' exact resolution — geometry is out (2026-09-30)

Full result in `results/mvtec_ad2/patchcore_ref/walnuts.md`. **The best localisation in the grid**: `au_pro_030`
averages 0.7595 (5.1× the 0.1487 random baseline) and `au_pro_005` averages 0.4479 (**18×** the
0.0249 baseline), with I-AUROC 0.802–0.862 — a spread of 0.060, the most stable detection of any
category run.

**Walnuts is 2448×2048. So is Wall Plugs, where the anchor is at or below chance in all six
conditions.** Same resolution, same 1.20:1 aspect, same square resize, 0.824 mean I-AUROC against
0.447. **Whatever separates a working anchor from a failing one is not the geometry of the frame**,
and the aspect-ratio mechanism registered on 2026-09-20 is now confined to Sheet Metal, where it
was registered and where it is still the only candidate.

### The direction-shift pattern has its third reading, and its first from a working anchor

`au_pro_005`, intensity conditions against direction shifts: Wall Plugs 0.1688/0.1723/0.1694
against 0.0985/0.0292/0.0753; Can 0.0036/0.0028/0.0077 against 0.0001/0.0000/0.0000; **Walnuts
0.4890/0.4953/0.4836 against 0.3670/0.4626/0.3896** (means 0.4893 vs 0.4064, same ordering at the
looser limit). The first two are categories whose anchor is broken; Walnuts is not, so the pattern
no longer rests only on failures.

**And on Walnuts detection is untouched:** `shift_1` has the best I-AUROC of the six (0.8622) and
the worst `au_pro_005` (0.3670). Direction costs precise localisation and nothing else.

**Still a tendency, not a rule — two of five contradict it.** Vial's worst condition is
`underexposed`, an intensity change; Fruit Jelly's `shift_1` is its best, though it has only one
shift. Report as three-of-five with the exceptions named.

### What this does to the rice/fabric question

The grid is **3–3**. Vial, Fruit Jelly and Walnuts give a usable anchor; Sheet Metal, Wall Plugs
and Can do not. `rice` and `fabric` are the last two, and both are 2448×2048 — the resolution that
just produced one of each. **So their outcome is genuinely unpredictable, which makes them the two
most informative categories left**, and dropping them would leave the study's central claim
balanced on a tie it chose not to break. That is a stronger argument than the selection-bias one
recorded on 2026-09-30, and it points the same way.

Also still true: the official evaluation-server metric averages per-category scores **over the
eight categories** (`docs/datasets-access.md`, secondary source, checkbox unconfirmed), so a
six-category study cannot compute it and M4 goes with the two. Confirm that on first login — it
costs nothing and it decides whether the exclusion is even available.

### Retraction: images-per-condition is not "climbing"

Yesterday's note called the count "trending up" after Wall Plugs' 25 and Can's 27. **Walnuts came
back to 25.** The series is 20, 19, 20, 25, 27, 25 — variable, with no direction. Corrected in
`docs/datasets-access.md`. `--max-bytes 12000000000` covers both 25 and 27 at 2448×2048 and is
what Walnuts ran with.

## Rice is at chance, and it ends the direction-shift pattern (2026-10-01)

Full result in `results/mvtec_ad2/patchcore_ref/rice.md`. **Detection at chance, localisation real**: I-AUROC
averages 0.519 (range 0.438–0.562) while `au_pro_030` is 2.4× and `au_pro_005` is **6.8×** their
measured random baselines. The grid is now **three usable anchors against four unusable**.

**Rice has the coarsest detection reading of the eight**: 42 normal images over six conditions is
**seven normals per condition**, 105 pairs, steps of 0.0095 — which is what the ±0.066 on two
conditions is made of. "At chance" is honest and low-resolution at the same time.

### The direction-shift pattern is dead, and the prediction that it would consolidate was wrong

Rice **inverts** it: `au_pro_005` intensity 0.1384 against shifts **0.2005**, and `au_pro_030`
0.3064 against **0.4008**. `shift_2` and `shift_3` are rice's two best conditions on both pixel
metrics, by a wide margin.

Tally over seven categories: **three for** (Wall Plugs, Can, Walnuts), **three against** (Vial —
worst condition is `underexposed`; Fruit Jelly — `shift_1` is its best; Rice — shifts are much
better). Sheet Metal's anchor measures nothing, so it does not vote. **Not writable as a finding.**

The 2026-09-30 handoff said a fourth reading "would make it writable in §6.1 as a tendency with
named exceptions". The fourth reading refuted it. That is the whole value of having written the
prediction down before the run, and the §6.1 comment in the paper has been corrected rather than
extended.

### What seven categories DO show, and it corrects an earlier reading of our own

Per-category means, pixel metrics as multiples of their random baselines:

| category | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| Vial | 0.910 | 5.7× | 22.9× |
| Walnuts | 0.824 | 5.1× | 18.0× |
| Fruit Jelly | 0.783 | 4.3× | 15.5× |
| Rice | 0.519 | 2.4× | 6.8× |
| Sheet Metal | 0.510 | 1.7× | 2.7× |
| Wall Plugs | 0.447 | 2.8× | 4.8× |
| Can | 0.388 | 2.0× | 0.1× |

**Ordering the seven by I-AUROC and by `au_pro_005` gives the same sequence apart from one adjacent
transposition** (Sheet Metal ↔ Wall Plugs). Where the anchor localises well it detects well.

Wall Plugs' and Can's reports describe detection failing while localisation survives — true inside
those categories, and it was starting to read as "the image-level score is the weak link". **It is
not, across categories: the two metrics move together.** The dissociation this study has evidence
for is between **lighting conditions within a category**, not between the metrics as such. Both
reports stand as written; the generalisation does not, and is corrected here and in §6.2.

## WinCLIP Phase A.2–A.3 — anomalib's ensemble is NOT the paper's: 21 templates, one duplicated (2026-10-03)

**Environment of this session** (notebook cell 0.3, Colab, 2026-10-03): `anomalib 2.6.0`,
`open-clip-torch 2.24.0` (anomalib's `[clip]` extra asks for the range `<2.26.1,>=2.23.0`),
`torch 2.11.0+cu130`, Python 3.13. Cell 1.1 reproduced the 2026-10-02 A.1 reading exactly
(signature and `Resize 240 BICUBIC` + CLIP `Normalize`), on a second VM.

**How these were read.** Cell 1.2's `dir()` loop crashed on `WinClipModel.patch_embeddings`, a
property that raises `RuntimeError` when empty (`getattr`'s default only catches
`AttributeError`). Instead of iterating, the installed source was read directly: the
`anomalib-2.6.0-py3-none-any.whl` from PyPI, files `anomalib/models/image/winclip/prompting.py`
and `torch_model.py`. The paper side is a direct read of arXiv:2303.14814v1, Figure 6, p.12.

**All three findings below were then OBSERVED on the Colab runtime** (same session, rewritten
cells 1.2–1.4): 1.2 printed `templates 21 | distinct 20`, `prompts normal 147 (distinct 140) |
anomaly 84 (distinct 80)`, the duplicate and both missing templates as listed below; 1.3 printed
`inner.class_name = None` after construction and `text_embeddings (2, 640)` after
`inner.setup('vial')` — D = 640, ViT-B/16+'s embedding width; 1.4 printed `PRETRAINED =
'laion400m_e31'`, `TEMPERATURE = 0.07`, and open_clip's tags for the architecture as
`['laion400m_e31', 'laion400m_e32']`. **Phase 1 of the notebook is closed.**

### 1. The prompt ensemble: state words match, templates do not

| | paper (Fig. 6) | anomalib 2.6.0 (`prompting.py`) |
|---|---|---|
| normal state words | 7 | 7 — identical strings |
| anomaly state words | 4 | 4 — identical strings |
| templates | 22 | **21 entries, 20 distinct** |
| normal prompts | 154 | **147** (140 distinct) |
| anomaly prompts | 88 | **84** (80 distinct) |

The two paper templates anomalib does not have: **`"a cropped photo of a [c]."`** and **`"a jpeg
corrupted photo of a [c]."`**. The second appears to have been typed as `"... of the {}."`, so
`"a jpeg corrupted photo of the {}."` is in the list **twice** and carries double weight in the
mean (`_collect_text_embeddings` averages all prompt embeddings per class into one vector, so
`text_embeddings` is `(2, D)`; a duplicate is not harmless).

**This is the plan's pre-registered priority-2 risk #1, now a measured fact, found before any
score exists.** It is small — 2 of 22 templates, both near-synonyms of templates that are present
— so it predicts a small shift, not a gate failure. That prediction is written here so it can be
held to: **if a gate misses by more than ~1 point, the template difference is not a sufficient
explanation on its own.** Whether to run anomalib as shipped or patch `TEMPLATES` to the paper's
22 is a protocol decision (§3: priority-2 implementation vs. verbatim paper prompts), NOT taken
here.

### 2. The weights: `laion400m_e31`

`torch_model.py:54-55`: `BACKBONE = "ViT-B-16-plus-240"`, `PRETRAINED = "laion400m_e31"`,
loaded by `open_clip.create_model_and_transforms(BACKBONE, pretrained=PRETRAINED)` (834 MB
download observed in cell 1.2). The paper says "LAION-400M based CLIP with ViT-B/16+" without an
epoch; open_clip ships both `laion400m_e31` and `laion400m_e32` for this architecture. So
anomalib's choice is **consistent with** the paper but not shown to be **the** paper's checkpoint.
`TEMPERATURE = 0.07`.

### 3. ⚠️ Phase B trap: `WinClip(class_name=...)` does NOT build the text embeddings

The Lightning `WinClip.__init__` stores `class_name` but constructs `WinClipModel()` without it,
so after construction `m.model.class_name is None` (observed in cell 1.2) and `text_embeddings`
raises. The embeddings are built only by `WinClip.setup(stage)` (the Lightning hook, which calls
`self.model.setup(self.class_name, ref_images)`) or by calling `m.model.setup(class_name)`
directly. **`winclip_backend.py` must do one of the two, per category**, because the class name
is the prompt noun — and a backend that forgot would not fail silently only because
`text_embeddings` raises. Note also that `_get_class_name` falls back to `"object"` when no name
is given.

## WinCLIP Phase A.1 — VERIFIED 2026-10-02, and the "256x256" claim is PatchCore's alone

Read off the installed anomalib 2.6.0 (`inspect.signature` + `WinClip.configure_pre_processor()`),
on a Colab T4 with torch 2.11.0+cu130.

**The constructor matches the plan's expectations exactly**, so nothing in Phase B has to adapt:

    WinClip(class_name: str | None = None, k_shot: int = 0, scales: tuple = (2, 3),
            few_shot_source=None, pre_processor=True, post_processor=True,
            evaluator=True, visualizer=True)

`k_shot=0` is the zero-shot default; `class_name` and `scales` are accepted; `scales=(2, 3)` is the
paper's pair of window scales. `few_shot_source` is WinCLIP+ and out of scope (§ this plan).

**The pre-processor, which is the load-bearing part:**

    Resize(size=[240, 240], interpolation=BICUBIC, antialias=True)
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
              std=[0.26862954, 0.26130258, 0.27577711])

Three findings, in order of what they change:

**1. 240x240 BICUBIC, not PatchCore's 256x256 BILINEAR — and it AGREES with the paper.** The paper
specifies ViT-B/16**+**, which is the 240px variant (`ViT-B-16-plus-240`). So anomalib's input
resolution matches the published configuration; that is one of this plan's priority-2 risks
retired rather than carried. The `Normalize` constants are CLIP's, not ImageNet's, which
corroborates that the whole pipeline is CLIP's own.

**2. The double-normalisation trap applies, and now explicitly.** `AnomalibModule.forward` runs
`self.pre_processor` unconditionally, so `winclip_backend.score()` must hand the model a **raw
tensor at native resolution** — no resize to 240, no CLIP normalisation. PatchCore's `score()`
did both for weeks while every test passed, because the operation is monotonic and preserves
ranking. Probe stage 11 is the standing guard for PatchCore; Phase B owes WinCLIP the equivalent.

**3. ⚠️ THE PAPER'S PRE-PROCESSING CLAIM IS PER-METHOD AND IS CURRENTLY WRITTEN AS GENERAL.**
`sections/04-benchmark-design.tex` §Metrics and the Threats-to-Validity comment in
`sections/06-analysis.tex` both say "a fixed square `Resize([256, 256])`". **That is PatchCore's.**
WinCLIP squashes to 240. Consequences:

* The two methods see **different** amounts of compression on the same image, so nothing may imply
  they run under one pre-processing pipeline. Protocol §3's own note — that choosing a priority-2
  implementation also fixes the input pipeline, which is therefore part of a run's identity — is
  exactly this, and it now has two instances with different numbers.
* The aspect-ratio mechanism registered for Sheet Metal is stated at 256. At 240 the compression
  is marginally worse (4224x1056 to a square 240 is 17.6x horizontal against 4.4x vertical). If
  WinCLIP's Sheet Metal number is also at chance, that is **not** evidence for the mechanism —
  both methods squash, and Walnuts already showed squashing does not predict failure.
* Every WinCLIP report must carry its own `Resize([240, 240])` caveat. The runner's report writer
  already emits PatchCore's caveats only for `patchcore_ref` (fixed 2026-10-02 in review), so the
  240 line is a WinCLIP-specific addition to make when its first report is written.

## ⚠️ Colab's torch build has moved: cu128 → cu130 (observed 2026-10-02)

A WinCLIP Phase A session reported `torch 2.11.0+cu130`. **Every PatchCore number in this repo was
produced on `torch 2.11.0+cu128`** — same torch version, different CUDA build —
pinned in `configs/methods/patchcore_ref.yaml:6` from the Colab T4 of 2026-08-21. anomalib is
still 2.6.0 and the pin test passes, so nothing about the method configuration changed.

**For WinCLIP this is immaterial:** it is a new method and cu130 simply goes into its own
provenance.

**For the figures it is a live risk, and this note exists so it is not investigated as a bug.**
Spec §5's legitimacy check is: regenerate a category's maps, then compare `image_score` against
the committed shard **at full precision**; a match proves the regenerated maps are the ones that
produced the reported numbers. The determinism that justified that check was measured on
2026-09-29 — Fruit Jelly, same four decimals, nine days and two VMs apart — but **both of those
runs were on the same torch build.** A different CUDA build is an untested variable, and
cuDNN/cuBLAS kernel selection can change low-order bits without anything being wrong.

**So if a figure regeneration fails the full-precision comparison, check the torch build BEFORE
concluding anything.** Three outcomes and what each means:

1. **Scores match exactly** — the check passes as designed, and it additionally establishes
   reproducibility across CUDA builds, which is a stronger claim than the study currently makes.
2. **Scores differ in the last decimals only** — environmental, not a defect. The figure is still
   of the same model on the same data, and its caption has to say the maps were regenerated under
   a different CUDA build than the numbers. The alternative is relaxing the comparison to a
   stated tolerance, which weakens the check and should be a recorded decision, not a default.
3. **Scores differ materially** — that is not a build difference and needs the systematic path.

**It cannot be avoided by pinning:** Colab controls the torch build, cu128 is no longer what a
fresh runtime gives, and a figure run needs a GPU. Recorded rather than solved.

## M3 IS COMPLETE — all eight categories, and five anchors are unusable (2026-10-01)

Fabric finished the public-split grid for the full-shot anchor: eight categories, three seeds each,
each aggregated per lighting condition. Per-category means, pixel metrics as multiples of their
**measured** random baselines (0.1487 at the 30% FPR limit, 0.0249 at 5%; see
`tests/test_au_pro_random_baseline.py`):

| category | W×H | aspect | i_auroc | au_pro_030 | ×rnd | au_pro_005 | ×rnd | anchor |
|---|---|---|---|---|---|---|---|---|
| Vial | 1400×1900 | 0.74:1 | 0.910 | 0.8414 | 5.66 | 0.5714 | 22.95 | **usable** |
| Walnuts | 2448×2048 | 1.20:1 | 0.824 | 0.7595 | 5.11 | 0.4479 | 17.99 | **usable** |
| Fruit Jelly | 2100×1520 | 1.38:1 | 0.783 | 0.6394 | 4.30 | 0.3865 | 15.52 | **usable** |
| Fabric | 2448×2048 | 1.20:1 | 0.562 | 0.1460 | **0.98** | 0.0076 | **0.30** | no |
| Rice | 2448×2048 | 1.20:1 | 0.519 | 0.3536 | 2.38 | 0.1695 | 6.81 | no |
| Sheet Metal | 4224×1056 | 4.00:1 | 0.510 | 0.2523 | 1.70 | 0.0677 | 2.72 | no |
| Wall Plugs | 2448×2048 | 1.20:1 | 0.447 | 0.4194 | 2.82 | 0.1189 | 4.78 | no |
| Can | 2232×1024 | 2.18:1 | 0.388 | 0.2984 | 2.01 | 0.0024 | **0.10** | no |

### The four things the grid established

**1. Five of eight categories have no usable full-shot anchor.** That is the headline, and it is a
result rather than a caveat: on most of this benchmark there is no meaningful ceiling for a
zero-shot method to be measured against, which is the comparison the whole study is built on.

**2. Geometry explains none of it.** Four categories share 2448×2048 at 1.20:1 — Walnuts (the best
anchor in the grid), Fabric, Rice and Wall Plugs (three failures, in three different shapes).
The aspect-ratio mechanism registered on 2026-09-20 predicted Sheet Metal and remains the only
candidate for that category alone.

**3. The failures take four distinct forms, and no mechanism covers two of them.**
  * *Sheet Metal* — at chance on detection, weak localisation (1.70× / 2.72×).
  * *Wall Plugs, Rice* — detection at or below chance, localisation clearly above random
    (2.82×/4.78× and 2.38×/6.81×).
  * *Can* — detection well below chance, localisation 2× at the loose limit and **below random**
    at the strict one.
  * *Fabric* — the inverse: detection weakly above chance in all six conditions, localisation **at
    the random baseline** (0.98×).

**4. The two metrics agree strongly but not perfectly across categories.** Spearman ρ = **0.833**
over the eight. The top three are identical in both orderings; Fabric is 4th by detection and 7th
by localisation, Wall Plugs the reverse. **Both dissociation directions occur**, which is a
stronger case for reporting the metrics separately than either direction alone.

### Two claims of ours that the grid retracted

* **The intensity-versus-direction split** (recorded 2026-09-29, predicted on 2026-09-30 to become
  writable with a fourth reading). Rice inverted it; final tally three for, three against. Dead.
* **"The same order apart from one adjacent transposition"** (recorded 2026-10-01 on seven
  categories). Fabric shifted three places and took ρ from near-1 to 0.833. The weaker, true
  version is above.

Both were written down before the runs that refuted them, which is the only reason they could be
retracted rather than quietly dropped.

### The four held defects are now due

`src/` was deliberately frozen for the whole grid so that every category ran on a code tree with no
differences across its commits — the property the 2026-09-29 determinism check depended on. **That
constraint is now lifted.** In rough order of what they cost:

1. **The coreset size is provenance read off a tqdm bar.** Missed on five of eight categories, and
   reading the bar instead of the bank put a wrong figure in three reports.
   `PatchCoreRef` should stamp `memory_bank.shape[0]` the way it already stamps `seed` and
   `preprocess`.
2. **`write_report` stamps the writing machine's HEAD** rather than the shards' `commit` column, so
   a resumed run produces a report claiming a commit that did not compute its numbers. It should
   read the column and refuse a single-commit provenance line when the shards disagree.
3. **The guard's suggested recovery command** omits `--summarise-only` and derives its budget from
   available RAM instead of the requirement it just computed — it suggested 9 GB for a 10.03 GB
   peak.
4. **Nothing guards a run that produced shards and no report.** That is how Fruit Jelly's first run
   vanished for nine days.

### Where the reports live, changed 2026-10-01

The eight reports moved to `results/mvtec_ad2/patchcore_ref/<category>.md` and new runs write
shards to `results/mvtec_ad2/<method>/<category>/shards`. `summarise` loads every parquet in a
shard directory, so without the method in the path two methods would pool into one row — silently,
because the filenames already differ.

**Pointers to a report were repointed everywhere, including inside dated handoffs**, because a
reference to a file that moved is a broken link rather than a historical fact. What was *not*
rewritten is any statement about where a run **wrote at the time**: the 2026-09-20 handoff still
says the Vial run wrote to `results/mvtec_ad2/vial/`, and that is true of that run.

(An earlier version of this note claimed pre-2026-10-01 handoffs were left as written. They were
not — their report pointers were updated, and the claim was corrected on 2026-10-02 after review
caught the contradiction.)

### What M3 does NOT include

* **Fit times and coreset figures** for five of the eight categories, as above.
* **The `validation` split**, deliberately unscored: `run_evaluation` re-fits per call and the
  threshold rule's seed semantics are not pre-registered. M4 needs it, and needs every archive
  again — now a `wget` per category rather than an upload.
* **Any zero-shot number.** This is the anchor only. Five methods still have no GPU backend
  (WinCLIP, AnomalyCLIP, AdaCLIP, SAA+, the Qwen2.5-VL baseline), each with its own pre-registered
  gate, and the grid above has to be re-run for each one that passes.

## The archives can be fetched in-session, and the GPU ran out before Fabric (2026-10-01)

**The project's largest cost was an untested assumption.** This file has said since 2026-09-20 that
"no session can fetch them", because the archives sit behind mvtec.com's form. The form issues a
**direct `mydrive.ch` share link per archive, with no login, no cookies and no session state** — a
`HEAD` on Fabric's returned `HTTP 200`, `application/gzip`, `Content-Length: 10837826502`
(10.84 GB), no redirect to authentication. Procedure and caveats in `docs/datasets-access.md`.

Measured against what it replaces: Drive upload ran at ~7 Mbit/s on the author's connection — two
hours for Rice's 6.29 GB, over three projected for Fabric. The in-session download runs at
datacenter bandwidth. **It also converts M4's ~30 GB of re-upload into a `wget` per category**,
which is the larger consequence, since every archive is needed again for `test_private`,
`test_private_mixed` and the `validation` split M3 deliberately did not score.

### The mistake that cost the session, written down so it is not repeated

Fabric's archive was downloaded **on a GPU runtime**. Colab's free GPU allowance is spent by
wall-clock time on a GPU runtime, so ~11 GB of transfer burned the scarce resource on work needing
no accelerator, and the session hit the GPU limit before a single seed was fitted. Nothing was lost
— no shard was written, so there is nothing stale anywhere — but the allowance was.

**The sequence that avoids it:** CPU runtime → `wget` to the VM → copy to `MyDrive/mvtec_ad2/`, a
Google-internal transfer measured in minutes → **then** a GPU runtime, which only mounts Drive and
fits. Drive's free 15 GB holds Fabric's 10.84 GB only if the other archives are deleted first.

## Where to pick up (session handoff, 2026-10-01, after Rice) — fabric, the last one

**Updated the same day, after a session that ran out of GPU before fitting anything.** Fabric's
archive was pulled with `wget` on a GPU runtime, which works but spends the GPU allowance on a
transfer. **Do the download on a CPU runtime and park the archive in Drive first** — see the
2026-10-01 section above. Nothing was written, so nothing is stale: the category has never run.


**M3 is 7 of 8: three usable anchors, four unusable.** All seven are run and reported, three seeds
each, shards on Drive.

**Only `fabric` is left** (387 train, 2448×2048, **10 GB**) — the largest archive in the dataset
and, at ~7 Mbit/s measured on Rice's 6.29 GB in two hours, **more than three hours of upload**.

**Try the direct download first; this is the category where it pays for itself.** Copy the
archive's true URL out of Chrome (`chrome://downloads` → copy link address, or DevTools → Network →
Copy as cURL if it is session-gated), then in Colab:

    !wget --progress=dot:giga -O /content/fabric.tar.gz "<URL>"
    !ls -lh /content/fabric.tar.gz && tar -tzf /content/fabric.tar.gz | head -3

~10 GB and paths beginning `fabric/` means it worked; run with `--archives /content`. A small file
or a `tar` complaint means an error page came down instead, and the normal upload is the fallback —
two minutes lost, not three hours. **Mount Drive anyway**: the shards still go there and they are
what survives the session. If the cURL form is needed it carries mvtec session cookies, so clear
that cell afterwards.

**The run:** phase 0 → cell 1.1 → `!python scripts/run_mvtec_ad2.py --category fabric --archives
/content` (drop `--archives` if it came via Drive), then
`--summarise-only --max-bytes 12000000000`. Fit ~4.1m per seed, ~12m for three.

**What to capture:**

1. **Whether the anchor is usable.** This decides 4–4 or 3–5, and `sections/05-results.tex` has
   been waiting since 2026-09-29 for the Table 2 design that depends on it.
2. **The coreset line** — bar 39627, bank **39628**. Record the bank. Missed four categories
   running now.
3. **The fit times.**
4. **Images per condition**, for the last row of the table in `docs/datasets-access.md`.

### After fabric, M3 is closed and the four held defects come due

All four were deliberately left unfixed so that `src/` stayed frozen across the grid — the property
the 2026-09-29 determinism check depended on. With M3 finished that constraint lifts:

1. `write_report` stamps the writing machine's HEAD rather than the shards' `commit` column.
2. The guard's suggested recovery command omits `--summarise-only` and derives its budget from
   available RAM instead of the requirement it just computed.
3. The coreset size is provenance read off a tqdm bar; `PatchCoreRef` should stamp
   `memory_bank.shape[0]` the way it already stamps `seed` and `preprocess`.
4. Nothing guards against a run that produced shards and no report — how Fruit Jelly's first run
   vanished for nine days.

## Where to pick up (session handoff, 2026-09-30, after Walnuts) — rice, then fabric

**M3 is 6 of 8, and the grid is 3–3 on whether the anchor is usable.** All six are run and
reported, three seeds each, shards on Drive.

**Next: `rice` (313 train, 2448×2048, 6.29 GB), then `fabric` (387, 2448×2048, 10 GB).** These two
decide whether a majority of the benchmark has a usable full-shot anchor, so they are the most
informative categories left rather than the most expendable.

**Try the direct download first.** Uploading through Drive is the project's real bottleneck and it
may not be necessary: copy the archive's true URL out of Chrome (`chrome://downloads` → copy link
address, or DevTools → Network → Copy as cURL if it is session-gated), then in Colab

    !wget --progress=dot:giga -O /content/rice.tar.gz "<URL>"
    !ls -lh /content/rice.tar.gz && tar -tzf /content/rice.tar.gz | head -3

and run with `--archives /content`. The VM has ~100 GB of disk, the download runs at datacenter
bandwidth, and Drive's 15 GB stops mattering. **Shards still go to Drive** — mount it first, they
are small and it is what survives the session. If the cURL form is needed, note that it carries
mvtec session cookies: do not leave that cell's contents anywhere persistent.

**The run:** phase 0 → cell 1.1 → `!python scripts/run_mvtec_ad2.py --category rice --archives
/content`, then the summary with `--summarise-only --max-bytes 12000000000`. Fit ~2.7m per seed for
rice, ~4.1m for fabric.

**What to capture — the first two have now been missed three categories running:**

1. **The coreset line.** Bar reads one less than the bank. Rice: bar 32050, bank **32051**. Fabric:
   bar 39627, bank **39628**. Record the bank.
2. **The fit times.**
3. **Whether `au_pro_005` puts the intensity conditions above the direction shifts.** Three of five
   do so far. A fourth would make it writable in §6.1 as a tendency with named exceptions.
4. **Whether the anchor is usable.** This is the tiebreaker, and `05-results.tex` has been waiting
   since 2026-09-29 for the Table 2 design decision that depends on the answer.

Older handoff (2026-09-30, after Can) follows.

## Where to pick up (session handoff, 2026-09-30, after Can) — walnuts

**M3 is 5 of 8.** Vial, Sheet Metal, Fruit Jelly, Wall Plugs and Can are run and reported, three
seeds each. **Three of the five have an unusable anchor** — a fact Table 2 now has to be designed
around rather than footnoted; see `sections/05-results.tex` in the paper repo.

**Next: `walnuts` (432 train, 2448×2048, 5.88 GB)**, by upload size. Then `rice` (6.29 GB) and
`fabric` (10 GB) last.

**The guard will trip, and 11 GB may not be enough.** Budget for **27** images per condition, not
25: at 2448×2048 that is **10.83 GB**, against the 11 GB that carried Wall Plugs at 25 images
(10.03 GB). Use `--summarise-only --max-bytes 12000000000`, and read the real count off the layout
check first — it prints seconds after extraction, before any fit.

**Budget:** 5.1m fit per seed, ~15m for three. The longest fit of the grid.

**What to capture:**

1. **The coreset line** as a cross-check of the train count. The bar will read **44235**; the bank
   is **44236** = `floor(432 × 102.4)`. Record the bank.
2. **The fit times**, which have gone unrecorded for two categories running.
3. **Whether `au_pro_005` separates intensity conditions from direction shifts**, as it did on Wall
   Plugs and Can. A third reading in the same direction is what would make it writable in §6.1.
4. **If the anchor fails here too**, that is four of six, and the study's comparison design needs
   the decision that `05-results.tex` has been asking for since 2026-09-29 — before `rice` and
   `fabric` are uploaded, because it may change what is worth running.

Older handoff (2026-09-29, after Wall Plugs; reordered 2026-09-30) follows.

## Where to pick up (session handoff, 2026-09-29, after Wall Plugs; reordered 2026-09-30) — can

**M3 is 4 of 8.** Vial, Sheet Metal, Fruit Jelly and Wall Plugs are run and reported, three seeds
each. Shards for all four are on Drive.

### The order changed on 2026-09-30, and the reason is worth keeping

The remaining four were ordered smallest-train-set-first, so that a cost surprise would arrive
early and cheap. Four categories in, **that is the wrong axis**: the fit costs 2.7–5.1 minutes per
seed, 8–15 minutes for three, and it has never been the bottleneck. The bottleneck is the manual
upload, which scales with the archive, and by archive the order is different:

| order by train count (old) | order by upload (new) |
|---|---|
| rice 313, fabric 387, can 412, walnuts 432 | can 2.65 GB, walnuts 5.88, rice 6.29, fabric 10 |

`can` is both the cheapest upload and the only one of the four that does **not** trip the memory
guard, so it is the shortest possible session. `fabric` stays last regardless: 10 GB is a third of
what remains to upload, against Drive's free 15 GB.

**Nothing about this affects a number.** Category order enters no metric and each is aggregated
separately; this is purely which upload costs least first.

**Next: `can` (412 train, 2232×1024, 2.65 GB).** Upload its archive to `MyDrive/mvtec_ad2/` and
delete `wallplugs.tar.gz` once its shards are confirmed. Then phase 0 → cell 1.1 → cell 50 with
`CATEGORY = "can"`. Fit ~4.6m per seed, ~14m for three — the longest fit of the grid so far, and
still not the slow part.

**The guard should NOT trip on this one.** 2232×1024 is 2,285,568 pixels per image: 4.57 GB at 25
images per condition, 3.66 GB at 20, both under the 6 GB limit. It is the only remaining category
where the plain `!python scripts/run_mvtec_ad2.py --category can` should run start to finish.
`rice`, `walnuts` and `fabric` are all 2448×2048 and all need
`--summarise-only --max-bytes 11000000000` — budget for 25 images per condition, not 20, and do
not follow the command the script prints (see the defect above). Read the real
images-per-condition off the layout check, which runs seconds after extraction and before any fit.

**Three things to capture, and the first two have been missed twice now:**

1. ~~**`model.memory_bank.shape[0]`**, to settle the coreset off-by-one.~~ **DONE 2026-09-30:
   the bank is `floor(N × 102.4)` and the progress bar shows one less.** See the section above.
2. **The coreset line** (`Selecting Coreset Indices`) from the fit, now as a plain cross-check of
   the train count rather than an open question. The bar will read **42187** for can; the bank is
   **42188** = `floor(412 × 102.4)`. For the rest: rice bar 32050 / bank 32051, fabric 39627 /
   39628, walnuts 44235 / 44236. **Record the bank figure, not the bar's** — reading the bar as the
   bank is what put a wrong number in three reports.
3. **Whether AU-PRO degrades under the direction shifts.** Wall Plugs separates cleanly at strict
   FPR: the three intensity conditions inside 0.0035 of each other, the three `shift_*` a mean
   2.5× worse. Vial degraded, Fruit Jelly did not, Wall Plugs does at `au_pro_005` only. A fourth
   reading is what decides whether §6.1 can say anything general.

Older handoff (2026-09-29, after Fruit Jelly) follows.

## Where to pick up (session handoff, 2026-09-29, after Fruit Jelly) — wallplugs

**M3 is 3 of 8.** Vial, Sheet Metal and Fruit Jelly are run and reported, three seeds each.
Fruit Jelly's shards are on Drive; its first, unreported run's shards were deleted from both the
VM and Drive on 2026-09-29 and must not come back.

**Next: `wallplugs` (293 train, 2448×2048, 2.05 GB).** Upload `wallplugs.tar.gz` to
`MyDrive/mvtec_ad2/` first; `fruit_jelly.tar.gz` can be deleted once its shards are confirmed on
Drive. Then phase 0 → cell 1.1 → cell 50 with `CATEGORY = "wallplugs"`.

**Expect the memory guard to trip, unlike the three so far.** 2448×2048 at ~20 images per
condition is 8.02 GB against the 6 GB `pixel_metrics` guard. The shards are written before the
summary runs, so the recovery is the same one Sheet Metal used:
`--summarise-only --max-bytes 8000000000`. This is the expected path for `wallplugs`, `rice`,
`fabric` and `walnuts` — four of the five that remain. Only `can` (2232×1024, 3.66 GB) should pass
cleanly.

**Budget:** 2.3m fit per seed, ~7m for three, by the cost model — which has now been right three
times (Vial 2m19s, Sheet Metal 29s against 31s predicted, Fruit Jelly 1m51s against 1.9m).

**Two things to capture, beyond the usual lighting count and coreset size:**

1. **`model.memory_bank.shape[0]` after a fit**, to settle whether the coreset off-by-one is the
   bank or the progress bar. See the 2026-09-29 section above.
2. **Whether AU-PRO stays flat under lighting.** Fruit Jelly's did and Vial's did not. A third
   reading is what turns this from two anecdotes into something §6.2 can be written from.

Older handoff (2026-09-29, before Fruit Jelly ran) follows.

## Where to pick up (session handoff, 2026-09-29) — M3 continues, with Fruit Jelly

**M3 is 2 of 8.** Vial and Sheet Metal are run, three seeds each, reports committed
(`results/mvtec_ad2/patchcore_ref/vial.md`, `results/mvtec_ad2/patchcore_ref/sheet_metal.md`, commit `a350e04`). Shards are on
Drive at `MyDrive/mvtec_ad2_results/<category>/`. Nothing from that session is outstanding: the
runner now writes its own report file, restores shards from Drive, and survives missing maps.

**The two results do not tell the same story, and that is the finding so far.** Vial's anchor
works and degrades under lighting exactly as the dataset intends (0.95 `regular` → 0.87
`underexposed`). Sheet Metal's anchor is **at chance in every condition including `regular`** —
the pre-registered aspect-ratio mechanism's most extreme case, though causation is untested and
must not be claimed. Both outcomes are now recorded in the paper repo
(`docs/verified-literature-facts.md`, addendum 2026-09-29, plus comment blocks at the two sites in
§5 and §6 that owe them), so the provenance is current up to Sheet Metal.

### The prerequisite, and it is the only slow part

**Upload `fruit_jelly.tar.gz` (1.2 GB) to `MyDrive/mvtec_ad2/` before the session starts.** It is
behind mvtec.com's registration form, so no session can fetch it. Vial (0.77 GB) and Sheet Metal
can be deleted from Drive first — their shards are already safe under `mvtec_ad2_results/`.

### The session, in full

1. **Phase 0** entire, fresh runtime. Cell 0.1's `%cd` + `pip install -e .` is what makes
   `!python scripts/...` work at all; a subprocess does not inherit cell 1.1's `sys.path`.
2. **Cell 1.1** — hard sync.
3. **One typed line:** `!python scripts/run_mvtec_ad2.py --category fruit_jelly`

**Budget: well under half an hour.** The cost model puts the fit at 1.9m per seed, 5.7m for three
(263 train images, exponent 2.06 fitted on Vial and VisA's candle). Scoring 80 public-test images
three times sits on top.

**The memory guard should NOT trip, unlike Sheet Metal.** Fruit Jelly is 2100×1520, projected at
5.11 GB peak against the 6 GB `pixel_metrics` guard — but that projection assumes ~20 images per
lighting condition, and Fruit Jelly's condition count is **not measured**. If it trips, the run is
recoverable exactly as Sheet Metal's was: re-invoke with
`--summarise-only --max-bytes 8000000000`. The shards are already written by then; only the
aggregation is repeated.

### Three things to watch, in the order they will appear

1. **The lighting-condition count**, printed within seconds of extraction, before any fit. It
   fills the third row of the table in `docs/datasets-access.md`, which currently has Vial (7
   conditions, 20 images each) and Sheet Metal (6, 19). Fruit Jelly's `test_public` is 80 images
   (20 normal / 60 anomalous, Table 4), so ~20 per condition implies **4 conditions** — an
   inference from the near-constant, not a measurement. Record what the script actually prints.
2. **The coreset size, which is a live discrepancy in our own record.** Both existing reports
   write the formula as `floor(N × 102.4)` and both report a number **exactly one less** than it
   gives: Vial 29797 against 29798, Sheet Metal 14027 against 14028. For Fruit Jelly the formula
   gives 26931, so the pattern predicts **26930**. Either value resolves it — record which, then
   correct the formula in both reports if it is off by one. Separately, the coreset size still
   cross-checks the train count against Table 4, which is what it was there for.
3. **Per-condition granularity, for the report's caveats.** At 4 conditions Fruit Jelly gets 5
   normal and 15 anomalous per condition — 75 pairs, so I-AUROC moves in steps of 0.0133 and a
   ±0.0000 across seeds means the metric could not resolve a difference. Sheet Metal's report
   states this for 60 pairs; Fruit Jelly's needs its own version with the real counts.

### After Fruit Jelly

`wallplugs` (293 train), `rice` (313), `fabric` (387), `can` (412), `walnuts` (432) — smallest
first, so any cost surprise arrives early and cheap. **Four of those five are 2448×2048 and are
projected at 8.02 GB, over the guard**: `--summarise-only --max-bytes 8000000000` is the expected
path for `rice`, `fabric`, `wallplugs` and `walnuts`, not an exception. Only `can` (2232×1024,
3.66 GB) should pass cleanly. Each needs its archive uploaded by hand, one at a time.

### Still open, not blocking M3

- **What M2 means.** The README scopes it to every method's GPU backend; this file's ordered list
  says PatchCore's gate closes it. The checkbox is still deliberately unticked.
- **VisA's §6 debt.** Seed 0 alone is a first reading; seeds 1 and 2 are ~8 h more for a check
  that cannot change a verdict. A budget decision, still not decided.
- **VisA's released image dimensions**, for the square-resize hypothesis registered 2026-09-19.
  MVTec AD 2's are confirmed and *none is square*. One line against the data, into
  `datasets-access.md` first.
- **The confirming experiment for Sheet Metal** — the same category under an aspect-preserving
  transform. §7 permits it only as a separate, dated experiment whose outcome is recorded either
  way. Not scheduled, and deliberately not part of M3.

Older handoff (2026-09-20, the session that started M3) follows.

## Where to pick up (session handoff, 2026-09-20) — M3 starts, with Vial

**VisA is done and committed** (`results/reproduction/patchcore_visa.md`, 86.26 against a
published 92.4, non-gating). Shards are on Drive at `MyDrive/reproduction_visa/`. Nothing from
that session is outstanding.

**Tomorrow is the first REPORTABLE MVTec AD 2 number of the study.** PatchCore's gate passed, so
protocol §2 no longer blocks AD 2 results.

### The session, in full

1. **Phase 0** entire, fresh runtime. Still not optional: cell 0.1's `%cd` + `pip install -e .`
   is what makes `!python scripts/...` work at all (a subprocess does not inherit cell 1.1's
   `sys.path`).
2. **Cell 1.1** — hard sync.
3. **One typed line:** `!python scripts/run_mvtec_ad2.py --category vial`

**You do not need cell 22.** The script mounts Drive, fetches `MyDrive/mvtec_ad2/vial.tar.gz`,
extracts, verifies the layout with `prepare_data.py`, runs seeds 0/1/2, copies shards to
`MyDrive/mvtec_ad2_results/vial/` **after every seed**, and prints a per-lighting aggregation.
`vial.tar.gz` has been on Drive since 2026-09-16, so nothing needs uploading.

**Budget: well under an hour.** Vial's fit is ~2m20s per seed (291 train images, measured), so
~7 minutes of fitting plus scoring 140 public-test images three times.

**Expect seed 0 to re-run, and that is correct.** Phase 2's Vial shard lives at
`results/patchcore/vial/shards`, a different root, and it predates both the gate passing and the
`preprocess` column. It was never reportable. The new run writes to `results/mvtec_ad2/vial/` and
starts clean.

### After Vial

Order the remaining seven **smallest first**, so any cost surprise arrives early and cheap:
`sheet_metal` (137 train), `fruit_jelly` (263), `wallplugs` (293), `rice` (313), `fabric` (387),
`can` (412), `walnuts` (432). Each needs its archive uploaded to `MyDrive/mvtec_ad2/` once, by
hand — that is the real bottleneck, not the GPU. Drive's free 15 GB holds one large category at a
time, so delete an archive once its shards are safe.

**Two things to watch on the first category that is not Vial:** the coreset size printed in the
log gives the train count (`floor(N x 102.4)`), which cross-checks Table 4's figure for that
category; and the float16 overflow guard has only ever been exercised on Vial and VisA.

### Still open, not blocking M3

- **What M2 means.** The README scopes it to every method's GPU backend; this file's ordered list
  says PatchCore's gate closes it. The checkbox is still deliberately unticked.
- **VisA's §6 debt.** Seed 0 alone is a first reading; seeds 1 and 2 are ~8 h more for a check
  that cannot change a verdict. Flagged as a budget decision, not decided.
- **VisA's released image dimensions**, for the square-resize hypothesis registered 2026-09-19.
  MVTec AD 2's are now confirmed from Table 4 and *none is square*, which makes the question
  larger than VisA. One line against the data, recorded in `datasets-access.md` first.

Older handoff (2026-09-18, after the seeds session) follows.

## Where to pick up (session handoff, 2026-09-18, after the seeds session)

**PatchCore's reproduction gate PASSED** (see "THE GATE PASSED" above): 98.02 ± 0.07 against a
published 99.0, inside ±1.0 by 0.02 points, three seeds, report committed. Nothing about the
MVTec AD gate needs re-running.

**The one thing left from that session is VisA, and it did not fit.** The seeds took the session;
VisA is **~4 h** on its own (measured 2026-09-19; this line said ~2 h) and was not started. It is a *reported secondary check* that cannot fail
the method (§2 v0.2.12) — now a second independent reading of an implementation that has already
passed its real gate, rather than evidence for a pending decision.

**Session state: the three-seed shards were copied to `MyDrive/reproduction_3seed/`** — a new
folder, deliberately not `MyDrive/reproduction/`, which holds the 2026-09-16 seed-0 shards under
byte-identical filenames and would have been overwritten. The maps were not copied and are not
needed: the gate is image-level, and threshold calibration runs on MVTec AD 2, not on classic.

### MEASURED 2026-09-19: VisA costs ~4 h, not ~2 h — every "~2 h" below is wrong

From the live run, not from an estimate. **7 of 12 objects in 2 hours**, with the four pcb
objects (the other large ones) and pipe_fryum still to go.

**macaroni2's fit alone took 23m39s.** Its coreset printed 92159 on the sampler's bar, i.e. 92160
rows, i.e. `92160 / 102.4 = 900` training images — which also confirms the `anomalib` transform's
32x32 grid reached this object, the same per-category check every MVTec run got.

**The mechanism, so the number generalises instead of being a surprise twice.** Greedy coreset
selection is roughly *quadratic* in the number of training patches, and VisA's large objects have
**900** training images against MVTec AD classic's largest, hazelnut, at **391**. Hazelnut's fit
took 2m29s. So a single large VisA object costs about **ten times** the worst MVTec category, and
VisA has five or six of them. The earlier note that corrected "cheap relative to a 15-category
fit" to "roughly two hours" did not go far enough: it scaled by image count, and the cost does not
scale by image count.

**What this changes downstream, and it is not small.** §6 owes VisA three seeds like anything else
reported. At ~4 h per seed that is **~12 h of T4 time for one secondary check that cannot fail the
method** (§2 v0.2.12). That is now a budget decision worth making deliberately rather than
inheriting from a line that says "~4 h more". It is flagged, not decided.

**The per-object train counts are recovered, and the transform is verified per object.** Same
arithmetic as the MVTec runs — the sampler's bar totals `rows - 1`, and `floor(N x 102.4)` gives
the coreset size, so `N` is recoverable. All eight so far land on integers, which is the
per-category proof that the `anomalib` 32x32 grid reached each one rather than only the first:

| object | coreset rows | train images | fit |
|---|---|---|---|
| candle | 92160 | 900 | 23m39 |
| capsules | 55500 | 542 | 8m21 |
| cashew | 46080 | 450 | 5m39 |
| chewinggum | 46387 | 453 | 5m51 |
| fryum | 46080 | 450 | 5m39 |
| macaroni1 | 92160 | 900 | 23m41 |
| macaroni2 | 92160 | 900 | 23m39 |
| pcb1 | 92569 | 904 | 23m51 |

**5499 over eight objects**, against VisA 1-cls's documented 8659, so **3160** remain across
pcb2, pcb3, pcb4 and pipe_fryum — the record closes exactly when they land. Fit time through
pcb1: **2h00m**.

**The quadratic cost is now measured rather than inferred.** cashew and candle differ by exactly
2x in training images (450 vs 900) and by **4.19x** in fit time, where quadratic predicts 4.0.
It is also visible directly in the sampler's own rate: **135.6 it/s at 450 images against 64.9 at
900** — each greedy step scans a bank twice as large, so doubling the images quarters nothing and
quadruples the wall clock. That is the whole explanation for why VisA costs ~4 h and MVTec AD
classic's fifteen categories cost ~40 minutes.

**Operational, learned the same hour:** during a multi-hour run the finished objects live only on
the VM's disk, and Colab serializes cells, so cell 3.5 cannot protect them until the run ends.
`scripts/save_visa_to_drive.py` exists for that — interrupt after a `[N/12] done` line, cell 1.1,
one typed line, re-run 3.4. Cell 1.1 is safe to run mid-session: `git reset --hard` does not touch
untracked files and the `git clean` is scoped to `src`, so `data/visa` and the shards survive.

### Tomorrow's session is short, and shorter than the last two

VisA needs **no Drive upload and no cell 3.1**. `scripts/run_visa_secondary.py` fetches the 1.8 GB
archive itself from AWS Open Data (no registration, CC BY 4.0) and checks its byte-exact size
before starting, precisely so a two-hour run cannot begin on a truncated download.

1. **Phase 0** entire, fresh runtime. **It is not optional, and the reason is not obvious:**
   cell 1.1 adds `src` to the *kernel's* `sys.path`, and `!python scripts/...` starts a new
   interpreter that does not inherit it. What makes the script importable is cell 0.1's
   `%cd` + `pip install -e .`. Skipping straight to 1.1 ends in `No module named 'vlmab'`.
2. **Cell 1.1** — hard sync.
3. **Cell 3.4, one line:** `!python scripts/run_visa_secondary.py` (**~4 h — measured
   2026-09-19, see the section above; the ~2 h this line used to say was wrong**, resumable per
   object, so a drop costs one of twelve). It scores itself and writes
   `results/reproduction/patchcore_visa.md`.
4. **Cell 3.5, immediately after** — copies the report *and the shards* to
   `MyDrive/reproduction_visa/` and reprints the report under a banner to copy out. **New on
   2026-09-19**, because nothing in the notebook did this: the only git cell is 20, which
   commits the backend and belongs to a phase this session skips, and cell 1.1's
   `git reset --hard` deletes an uncommitted report. The shards go too, so the gate can be
   re-scored, or one object re-run, without repeating the two-hour fit.

**Reload the notebook from `master` first.** The 2026-09-18 session ran on a copy that predated
`b86e209` (2026-09-17) and therefore had no cell 3.4 at all — which cost nothing, because the
check is one typed line, but it is the second time a stale notebook copy has shown up. Cell 1.1
syncs the *repository*; it cannot sync the notebook the session is executing. **This matters more
than last time: cell 3.5 is new, and an old copy will not have it.**

**Pre-session verification, done 2026-09-19 on CPU.** Every signature `run_visa_secondary.py`
calls was checked against the installed code — `run_evaluation`'s kwargs, `VisA.SPLITS ==
("train", "test")` against the `split`/`fit_split` it passes, `PatchCoreRef.seed`/`.preprocess`,
the gate's flags, and `secondary_n_categories: 12` in the targets file. Seven tests now exercise
the script's `run()` and `main()` with injected fakes; before that, **no line of either had ever
executed** — only the module import and one constant were covered, which is the shape of the
cell 3b.3 defect at a much higher price. One more thing confirmed: cell 1.1's `git clean -qfd src`
touches `src` only, so `data/visa` and the shards survive a re-sync mid-session.

**One flag to remember for later, not for this session:** seeds 1 and 2 go into the *same*
results root, and the gate refuses a root that pools seeds. They need
`--no-score` (`python scripts/run_visa_secondary.py --seed 1 --no-score`), or a correct two-hour
run ends in exit code 2.

### After VisA, in order

- **Commit `results/reproduction/patchcore_visa.md`.** A result in a closed session's scrollback
  is the same as no result.
- **Phase 4 is then finished** — the config pin landed 2026-09-17, the notebook is committed, and
  what remains is confirming both CI jobs stay green.
- **Settle what M2 means.** The README scopes it to every method's GPU backend; item 1 of this
  file says PatchCore's gate closes it. The checkbox is deliberately unticked until that is
  decided; the two documents should then say the same thing.
- **M3 is unblocked for PatchCore** — the full MVTec AD 2 grid, one category at a time, the
  download/resume/shard unit. Mechanical, and the largest category is Fabric at 10 GB.
- **WinCLIP is next on methods**, and its targets are already read and pre-registered (two gates,
  91.8 and 78.1). So is AnomalyCLIP's (two gates, two checkpoints).

### The CPU track, which needs no GPU session and can run in parallel with anything

~~**AdaCLIP and SAA+ are the last two papers unread**~~ — **AdaCLIP was read 2026-09-19; SAA+
is the last one.** §2 requires each gate settled from the method's own paper *before* its Colab
run. The four read so far took roughly an hour each, and three of the four turned up something
that would otherwise have surfaced mid-session: PatchCore's gate was the wrong dataset entirely,
AnomalyCLIP needs a different checkpoint per gate, and AdaCLIP reports its numbers twice and
needed its own repo read to settle which pair is the target.

What to expect from each, written before reading them so it can be checked afterwards:

- ~~**AdaCLIP**~~ — **DONE 2026-09-19, and the written-beforehand expectation was right on both
  counts and short on a third.** "Expect a checkpoint question and an overlap audit to refine":
  both happened. "The paper may settle the contingency before Colab does": it did not — the
  *repo* did, and the paper by itself could not even settle which of its two reported settings is
  the target. Worth keeping as the record of what reading a paper alone is and is not good for.
- ~~**SAA+**~~ — **DONE 2026-09-19, and the written-beforehand expectation was right about the
  prompts and silent about the bigger thing.** "The realistic outcome is that no AD 2 category has
  a published prompt": correct, and now resolved to `0_of_8` on CPU rather than in Colab. What it
  did not anticipate is that **SAA+'s paper publishes no image-level number at all**, so the ±1.0
  gate §2 is written in has no primary source for this method. The decision that opens is written
  out in its own section above.

Order it after VisA only because VisA needs the live session and this does not.
- **VisA's own §6 debt:** seed 0 alone is a first reading, not a table entry. Seeds 1 and 2 are
  `--seed 1` / `--seed 2` on the same script, **~8 h more at the measured cost**, owed before
  the number appears anywhere — and worth weighing against what a non-gating secondary check is
  for. See the measured-cost section above.

Both repos clean and in sync with `origin/master`; bench: 452 tests green.

Older handoff (2026-09-17, after the phase 3b session) follows.

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
