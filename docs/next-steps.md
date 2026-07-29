# Next steps

Living map of where execution stands and what to do next. Milestones (M1–M6) live in the README;
this file is the operational view — which plans are written, which halves are executed, and the
order for the work that remains. **Every method now has a written plan** (2026-07-29: AdaCLIP and
SAA+ were the last two). What remains is either a **Colab/GPU** step, an **external** one, or the
CPU half of the two newest plans — which is subagent-executable today. The pre-existing
CPU-verifiable core is done, on `master`, and green on CI (Python 3.11 + 3.13).

## What is done (CPU, no GPU)

- **Evaluation core:** image/pixel metrics (P-AUROC, AU-PRO, SegF1), provenance, a crash-safe
  result store, a resumable runner with per-sample latency, per-category `fit` for full-shot
  methods, and lighting-grouped aggregation. Protocol frozen at **v0.2.7**.
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
4. **AdaCLIP CPU tasks (1–3) then Colab phases (A–D)** —
   `docs/superpowers/plans/2026-07-29-adaclip-zeroshot-adapter.md`. Zero-shot and, like AnomalyCLIP,
   **auxiliary-trained**, so it carries the same kind of overlap audit. Its CPU half is
   subagent-executable now; `src/vlmab/methods/adaclip.py` is still a stub until Task 1 lands. Watch:
   the plan pre-registers a contingency for the case where the repo publishes only one checkpoint —
   resolve it in Colab phase A.2 before scoring anything.
5. **SAA+ CPU tasks (1–4) then Colab phases (A–D)** —
   `docs/superpowers/plans/2026-07-29-saa-trainingfree-adapter.md`. Training-free (GroundingDINO + SAM
   cascade), so **no aux-training concern** — simpler than AdaCLIP/AnomalyCLIP on the audit side, but it
   carries three things they do not: a dated §3 amendment for its per-category prompts (taken verbatim
   from the repo, committed before any scoring), the repo's own mask-based anomaly map taken verbatim
   plus a granularity diagnostic, and a **measured cost probe** (phase C.2) in place of the untested
   "may not fit a 12h session" assumption — the runner is resumable, so the real question is total
   budget, not session length. `src/vlmab/methods/saa.py` is a stub until Task 1 lands. Run this plan
   **after** AdaCLIP's: AdaCLIP's Task 2 re-points the registry's unknown-method example to `saa`, and
   SAA+'s Task 3 closes that chain.
6. **M3 — full MVTec AD 2 grid.** Once each method's VisA gate passes, run the full public-test grid
   over all eight categories, one category at a time (the download/resume/shard unit). Mechanical.
7. **M4 — evaluation server.** Score the private split. Blocked on the registration below.
8. **M5 — efficiency pass** on a rented fixed instance (protocol §5: latency never from Colab).
9. **M6 — preprint.**

## External / parallel

- **MVTec evaluation-server registration** (only blocks M4, not M2/M3). The company-email request
  was sent 2026-07-23; **follow up on 2026-08-06** if unanswered, then — if still unanswered — open a
  dated protocol amendment limiting the study to the public split (`docs/datasets-access.md` has the
  decision rule written down).
- **Dataset downloads for the grid:** only Vial (0.77 GB) is on disk. The rest of MVTec AD 2 is
  fetched per category on Colab when M3 runs (largest: Fabric, 10 GB); never into `/tmp` (tmpfs).

## Integration points every Colab playbook leaves to the executor

These cannot be pre-written without the data/repo in front of you, and each playbook marks them:
- **The VisA loader** over VisA's split CSV (its format is VisA-specific).
- **The published-numbers table** for the ±1pt gate (`PUBLISHED_*_VISA_IAUROC`), from each method's
  own paper — record the exact source next to the table.
- **The pinned versions/commits and checkpoint shas**, recorded back into the method's config and,
  for AnomalyCLIP, into the overlap audit's blank record-fields.
