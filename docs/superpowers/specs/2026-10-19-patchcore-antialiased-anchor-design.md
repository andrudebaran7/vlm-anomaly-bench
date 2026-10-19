# PatchCore antialiased anchor — design (protocol v0.2.21)

Date: 2026-10-19. Status: draft, awaiting the author's approval.

## Why

Check (b) (notebook phase 3c, probe stage 14, 2026-10-19; recorded in `docs/next-steps.md`,
"MEASURED 2026-10-19 — check (b)") confirmed both hypotheses:

- **H1:** `PatchCoreBackend.score()` goes through `PreProcessor.forward`, which applies
  `export_transform` — anomalib's copy of `transform` with antialiasing disabled. The memory bank is
  built through `engine.train`, which applies `transform` with antialiasing. Every committed
  PatchCore number — the MVTec AD classic gate (98.02 ± 0.07, passed by 0.02), VisA (86.26), the
  `classic` run (93.41) and all eight MVTec AD 2 categories — queried an antialiased bank with
  aliased features.
- **H2:** under `preprocess="classic"` the export path also lacks the CenterCrop, so the bank was
  built from 224x224 crops and queried with the uncropped 256x256 frame.

Since 2026-10-19 the paper is C1 + C3, so the PatchCore anchor is what C1 rests on. The author chose
(2026-10-19) to add a corrected configuration rather than report the defective numbers with a caveat.

## The commitment, made before any corrected number exists (the author's)

1. **The corrected configuration is the anchor, unconditionally.** It is not compared with the old
   one to pick a winner (§7, "no best of N"). Whatever it scores is what C1 reports.
2. **If it fails the MVTec AD classic gate (99.0 ± 1.0 I-AUROC), the anchor is reported as
   non-reproducing**, flagged in every table per protocol §2 — exactly as WinCLIP was.
   **The eight MVTec AD 2 categories run anyway** (the author, 2026-10-19): unlike WinCLIP's grid,
   they are not dropped. Every MVTec AD 2 report then carries the non-reproducing flag. So gate and
   grid may share a session, and the grid does not wait on the gate's verdict. The report
   generator needs a path for that flag.
3. **The old numbers are kept, never overwritten.** They become evidence for C3: an undocumented
   export path in a widely used library silently changed an evaluation pipeline, and a gate passed
   on it. Reporting old and new side by side as two dated configurations is allowed; choosing
   between them by score is not.

## Goals

- Protocol §3 **v0.2.21**: PatchCore's `forward` resizes with antialiasing, by the same mechanism as
  WinCLIP's v0.2.20 (`forward` routed to `transform`). Dated Changelog entry citing the check (b)
  measurement. Must land **before** any corrected run.
- Backend change in `src/vlmab/methods/patchcore_backend.py`: after construction (and after
  `_apply_classic_preprocessing`, so the classic crop is carried too), route `forward` to
  `transform`, refusing to run if `transform` has no antialiasing `Resize` — reuse
  `use_antialiased_forward` from `winclip_backend.py` (where it lives afterwards is the builder's
  call; its error messages currently name WinCLIP's v0.2.20 resize and must be generalised).
- The backend exposes `resize_antialias = True` and **every shard's provenance records it**, so an
  old and a new shard can never be averaged or confused (same reasoning as `preprocess` per shard).
- `configs/methods/patchcore_ref.yaml` records the new setting; its existing exact-pin test stays.
- A standing guard: the probe must now show `forward == train path` for both pre-processings on the
  non-square 480x640 image (stage 14 currently PASSES when H1/H2 are *confirmed*; the corrected
  backend needs the opposite assertion — a new stage or a mode of 14, builder's call). This runs in
  the Colab session before the gate, as phase 3c did.
- **Runs (the author's scope, 2026-10-19): the MVTec AD classic gate and the eight MVTec AD 2
  categories. Nothing else.**
  - Gate: 15 categories x seeds 0/1/2, `preprocess=anomalib`, otherwise the pinned config
    unchanged; report via `scripts/reproduction_gate.py`.
  - MVTec AD 2: all 8 categories, seeds 0/1/2, `test_public`, same runner and aggregation as the
    2026-09/10 grid; the same pre-registered usable/unusable-anchor criterion (random baseline)
    re-applied.
- New result files beside the old ones (naming is the builder's call); the old reports get a dated
  header line pointing to check (b) and the new file, nothing else changed in them.
- Notebook cells for the session, in the existing phase style (VERIFY steps, "paste the output").

## Non-goals

- VisA secondary check and the `classic` re-run are **not** re-run (author's scope).
- No hyperparameter change: backbone, layers, coreset ratio, neighbours, resize 256 all as pinned.
- No evaluation-server submission; M4 stays later, and only the corrected config may ever be
  submitted.
- No WinCLIP work.

## Anti-goals (failure even if it "works")

- Any old result file or shard overwritten or deleted.
- Choosing between old and new by score, or re-running "until it passes".
- Claiming aliasing *explains* the five unusable MVTec AD 2 anchors. If the new grid differs, that
  is a reported comparison of two configurations, not a diagnosis.
- A corrected number reaching the paper or a plan before it is in its provenance/result file.

## Constraints

- anomalib 2.6.0 pinned; Colab T4; the user cannot paste from the terminal into Colab — fixes reach
  the session by pushing to the repo and re-running cell 1.1.
- CI never imports anomalib/torch; the new logic is tested on CPU with stand-ins, as
  `use_antialiased_forward` already is.
- Datasets come from Drive (`MyDrive/mvtec_ad2/<category>.tar.gz`); all 8 were run before, so the
  uploads should already exist — the session verifies this rather than assumes it.
- Feature branch, `--no-ff` merge, never commit to master.

## Paper side (after the runs, not part of this build)

The `descope-c1-c3-rewrite` branch's abstract and C1 cite anchor numbers; they are updated from the
new result files once they exist, and `docs/protocol-sync.md` gets the v0.2.21 re-check.

## Calls made by Claude while writing this (the author can overrule any)

- **Hard-wired, not a switch.** Like WinCLIP's v0.2.20, the corrected backend always routes
  `forward` to `transform`; there is no `resize_antialias: false` path. The old numbers stay
  reproducible from git history: the plan tags the last pre-v0.2.21 commit. The yaml records
  `resize_antialias: true` as documentation, and a test holds it there. Provenance's
  `resize_antialias` tells new shards (true) from old ones (field absent).
- **`classic` refuses to run under v0.2.21.** With `forward` routed to the cropped `transform`, its
  map would cover only the central 224 px and be stretched over the whole frame. Classic is out of
  scope, so the backend raises a clear error for `preprocess="classic"` instead of being half-fixed.
  Making it runnable again would be a separate, later decision.
- **Runtime:** any Colab runtime with anomalib 2.6.0. Phase 0 prints and records torch/CUDA/anomalib
  versions in every new result file, but does not halt if they differ from 2026-08-21. Any
  old-vs-new comparison in the paper carries that runtime difference as a stated caveat.
- **Old reports annotated:** all affected files get one dated header line pointing to check (b):
  the gate reports (`patchcore_mvtec_ad.md`, `_3seed.md`), `patchcore_mvtec_ad_classic.md`,
  `patchcore_ref_visa.md` and the eight `results/mvtec_ad2/patchcore_ref/*.md`. Re-run files point
  to their replacement. VisA and classic say "affected, not re-run". The paper may cite their old
  numbers only as old-configuration evidence for C3, never as anchor numbers.

## Left to the builder

File/stage naming, where the shared helper lives, notebook cell layout, how the provenance field is
plumbed through the runner, test structure.
