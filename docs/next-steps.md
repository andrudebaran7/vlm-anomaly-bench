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
  methods, and lighting-grouped aggregation. Protocol frozen at **v0.2.11**.
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
   2. **Phase 1.2 smoke test**, confirmed green end to end. `fit` reached a 2048-vector memory bank
      before the last two fixes (tensor input, device after fit) landed; the assertion that the
      injected defect outscores a normal image has not been seen pass since.
   3. **Phase 2** — Vial end to end through the existing runner (notebook cells 21–26).
   4. **Phase 3 — the VisA ±1pt gate.** Deliberately not in the notebook: it needs the VisA loader
      and the `PUBLISHED_VISA_IAUROC` table (with its exact source recorded next to it), both listed
      under *Integration points* below. Writes `results/reproduction/patchcore_visa.md`.
   5. **Phase 4 — freeze provenance.** `configs/methods/patchcore_ref.yaml` still carries the
      `anomalib_version: ">=1.1"` range with "record the resolved version" next to it; the resolved
      version is **2.6.0** (Colab T4, torch 2.11.0+cu128, 2026-08-21). Commit the notebook, confirm
      both CI jobs stay green.
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

## The PatchCore Colab session (2026-08-21) — what it settled

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

## Integration points every Colab playbook leaves to the executor

These cannot be pre-written without the data/repo in front of you, and each playbook marks them:
- **The VisA loader** over VisA's split CSV (its format is VisA-specific).
- **The published-numbers table** for the ±1pt gate (`PUBLISHED_*_VISA_IAUROC`), from each method's
  own paper — record the exact source next to the table.
- **The pinned versions/commits and checkpoint shas**, recorded back into the method's config and,
  for AnomalyCLIP, into the overlap audit's blank record-fields.

## Where to pick up (session handoff, 2026-08-26)

**Both repos are clean and in sync with `origin/master`** — nothing is waiting to be pushed, which
was the open item in the previous handoff. Bench: **313 tests green** (~7 s). Paper: `master` at
`ef39c4b`, abstract and §1–§3 written and reviewed, §4–§6 stubs awaiting M3.

**The next move is the second Colab session**, and it resumes mid-phase rather than at the top:

1. Run notebook cell 1.1 to sync, then **probe stage 10** — the only open correctness question
   (double preprocessing, see step 1 of the ordered list). Do it before anything is scored: if the
   model normalises internally, every number produced until now is quietly wrong.
2. Then the 1.2 smoke test, phase 2 on Vial, and the VisA gate.

Fixes reach that session by being **pushed to `master`** — cell 1.1 hard-resets to `origin/master`
and prints what it synced. Push before asking for a re-run.

Two items only the author can close, both flagged in the paper's `references.bib` as `\todo` that
render as red text in the printed bibliography:
- the DOI for `duarte2026survey` (the author's own survey), and
- the full author list for `mllmzsad`.

One standing rule this project learned the hard way, worth restating: **a fact verified but not
recorded is a fact the next person cannot use.** Three times a claim was read from a primary source
and written straight into a task brief without landing in `verified-literature-facts.md`, and each
time the work correctly stalled until it was recorded. Record first, then write.
