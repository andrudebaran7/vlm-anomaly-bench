# Next steps

Living map of where execution stands and what to do next. Milestones (M1–M6) live in the README;
this file is the operational view — which plans are written, which halves are executed, and the
order for the work that remains. **Every method now has a written plan** (2026-07-29: AdaCLIP and
SAA+ were the last two). AdaCLIP's and SAA+'s CPU halves have since landed and are registered. What
remains is either a **Colab/GPU** step or an **external** one. The pre-existing CPU-verifiable core
is done, on `master`, and green on CI (Python 3.11 + 3.13).

## What is done (CPU, no GPU)

- **Evaluation core:** image/pixel metrics (P-AUROC, AU-PRO, SegF1), provenance, a crash-safe
  result store, a resumable runner with per-sample latency, per-category `fit` for full-shot
  methods, and lighting-grouped aggregation. Protocol frozen at **v0.2.9**.
- **MVTec AD 2 data path:** loader (verified against the real Vial archive), layout verification
  (`scripts/prepare_data.py`), and the run_eval CLI. Native-resolution, raw-scale maps (v0.2.6).
- **Method adapters — CPU halves done and registered** (each wraps an injectable backend; without
  a backend `prepare()` raises `MethodNotRunnable`, so the CLI never pretends it ran):
  - `intensity_baseline` — the dependency-free floor (fully working, no backend needed).
  - `mllm_qwen` — Qwen2.5-VL-3B; the response parser is complete, the model call is the injectable seam.
  - `patchcore_ref` — full-shot anchor; the anomalib backend is the seam.
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

1. **PatchCore anomalib backend** — `docs/superpowers/plans/2026-07-24-patchcore-anomalib-backend-colab.md`.
   The full-shot anchor; closes M2 once its VisA ±1pt gate passes. Do this first: it is the ceiling
   every zero-shot number is measured against, and anomalib's API is the most stable.
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
7. **M4 — evaluation server.** Score the private split. **Unblocked** — access granted 2026-07-30
   (`docs/datasets-access.md`). First action on first login is to read the server's submission docs
   for the official metric names and definitions: protocol §4 adopts them as the server defines them
   and still marks them unverified, so that reading is a §4 dependency. Also confirm the submission
   format and any attempt limit *before* spending a submission, since §7 commits to one per method,
   final.
8. **M5 — efficiency pass** on a rented fixed instance (protocol §5: latency never from Colab).
9. **M6 — preprint.**

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
