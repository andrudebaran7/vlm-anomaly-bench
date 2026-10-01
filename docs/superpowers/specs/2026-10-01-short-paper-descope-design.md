# De-scoping the study to a short paper — design

**Date:** 2026-10-01
**Status:** design, approved in conversation section by section; awaiting review of this document
**Supersedes:** nothing. Amends the scope the README's M1–M6 and `protocol.md` v0.2.17 describe.
**Protocol amendment this implies:** v0.2.18, **§3** (method list) and **§5** (M5 marked not exercised), drafted in section 3. §2's gates are deliberately untouched.

## Why

M3 closed on 2026-10-01 with the full-shot anchor measured on all eight MVTec AD 2 categories.
The study as planned needs five zero-shot methods to have GPU backends, each with its own
pre-registered gate, and the grid re-run for each — plus M4 and M5. None of the five backends
exists. The author's constraint, stated 2026-10-01: a short honest paper is an acceptable
outcome, one more round of dataset movement and GPU work is available, and the scope should not
extend past that.

This document fixes what the paper claims, what remains to be built, what is cut, and how each
piece is verified.

## 1. What the paper claims

A frontier benchmark (MVTec AD 2) measured under a pre-registered protocol with **two methods** —
a full-shot anchor (PatchCore via anomalib 2.6.0) and **one** zero-shot method (WinCLIP) — over
all eight categories, aggregated per lighting condition, on the public and private splits. No
efficiency axis.

### C1 — the full-shot anchor is unusable on five of eight categories

Measured, three seeds each, `test_public` (`results/mvtec_ad2/*.md`):

| category | aspect | i_auroc | au_pro_030 (×rnd) | au_pro_005 (×rnd) | anchor |
|---|---|---|---|---|---|
| Vial | 0.74:1 | 0.910 | 0.8414 (5.66) | 0.5714 (22.95) | usable |
| Walnuts | 1.20:1 | 0.824 | 0.7595 (5.11) | 0.4479 (17.99) | usable |
| Fruit Jelly | 1.38:1 | 0.783 | 0.6394 (4.30) | 0.3865 (15.52) | usable |
| Fabric | 1.20:1 | 0.562 | 0.1460 (0.98) | 0.0076 (0.30) | no |
| Rice | 1.20:1 | 0.519 | 0.3536 (2.38) | 0.1695 (6.81) | no |
| Sheet Metal | 4.00:1 | 0.510 | 0.2523 (1.70) | 0.0677 (2.72) | no |
| Wall Plugs | 1.20:1 | 0.447 | 0.4194 (2.82) | 0.1189 (4.78) | no |
| Can | 2.18:1 | 0.388 | 0.2984 (2.01) | 0.0024 (0.10) | no |

Four distinct failure modes, with no mechanism shown to cover two of them; frame geometry refuted
as the variable (four categories share 2448×2048 and the anchor works on one). The statement is
only expressible because the AU-PRO random baseline was measured — 0.1487 at the 0.3 FPR limit,
0.0249 at 0.05, pinned by `tests/test_au_pro_random_baseline.py`. Without it `au_pro_030 = 0.1460`
cannot be read at all.

### C2 — one zero-shot method against the anchor, and where that comparison is undefined

WinCLIP on the same eight categories, with C1 folded in: the full-shot gap exists on three
categories and is **undefined on five**. What was planned as a comparison table becomes a claim
about which comparisons the benchmark supports.

### C3 — why no zero-shot method reports frontier numbers (already written, §6.5)

Per-benchmark dependencies; reproduction criteria that do not survive reading the papers;
artefacts diverging from publications; silent and plausible failure modes; validation cost not
proportional to benchmark size. Sourced from papers and official repositories at recorded commits,
not from runs.

### The honesty obligation in the text

`sections/01-abstract.tex` through `03-related-work.tex` promise five methods. They are rewritten
to promise one plus the anchor. **§6.5's claims about the other four stay**, reframed from "methods
we evaluated" to "methods whose artefacts we analysed and could not evaluate, for reasons that are
part of the result". That is not a retraction; it is a description of what was done.

WinCLIP's selection is justified by a dated criterion: the only one of the five with unambiguous
published targets at the configuration this repo runs, recorded 2026-09-18 in
`configs/reproduction/winclip.yaml` and the paper repo's provenance file, **before any zero-shot
run existed**. Not result-driven, and the date is the proof.

## 2. What must be built and run

WinCLIP's CPU half is already done and registered: `src/vlmab/methods/winclip.py` → `WinClipRef`,
registry key `"winclip"`, with `configs/methods/winclip.yaml` and targets in
`configs/reproduction/winclip.yaml`. Only the GPU backend is missing.

But the session runners are PatchCore-specific. `scripts/run_eval.py` is generic (registry +
`--method`); `run_mvtec_ad2.py` and `run_visa_secondary.py` import `PatchCoreBackend` directly, and
**the MVTec AD classic gate has no script at all** — PatchCore's ran from notebook cells 3.2/3b.1.

### A — CPU batch: make the session runners method-generic, and close three defects

The only item that needs no GPU.

1. `run_mvtec_ad2.py --method {patchcore_ref,winclip}`. Drive fetch, `restore_from_drive`, the
   pixel-metric guard, per-seed shard copying and `write_report` are already method-agnostic; only
   backend construction is hardcoded. This is where the machinery that took weeks to settle lives,
   so it is generalised rather than duplicated.
2. Generalise `run_visa_secondary.py` the same way, and **write the MVTec AD classic gate path as a
   script**. WinCLIP needs both gates; PatchCore's classic runs were done by hand.
3. Seed count comes from the method: three where stochasticity exists (§6), one where it is
   measured absent.
4. Defects #2 (`write_report` must take the commit from the shards' `commit` column and refuse a
   single-commit line when they disagree), #3 (the guard's recovery suggestion must include
   `--summarise-only` and derive its budget from the computed requirement), #4 (a guard against a
   run that produced shards and no report). Defect #1 (stamp `memory_bank.shape[0]` into
   provenance) is PatchCore-only and M3 is closed, so it is included only because it is small.

**Constraint from section 5: A must not touch the scoring path.** Only the session runners. If
`patchcore_backend.py`, `run_evaluation` or the metrics change, the regenerated maps the figures
need are no longer the maps that produced the reported numbers.

Touching `src/` moves the commit, which is now acceptable: M3's numbers are frozen in committed
reports, and §6 already treats different methods as different configurations, so cross-method
comparison never required a shared code commit.

### B — GPU: the WinCLIP backend (plan phase A)

`src/vlmab/methods/winclip_backend.py`, wrapping anomalib's WinClip, tracked so a Colab cell can
pull it, with torch imported lazily so CI never touches it — the pattern `patchcore_backend.py`
established.

Three pre-registered checks run **before any scoring**:

1. **The prompt ensemble is a count.** The paper's Figure 6 gives 7 normal state words, 4 anomaly
   state words and 22 templates: **154 normal and 88 anomaly prompts**. Compare against what
   anomalib builds.
2. **Which CLIP weights anomalib resolves to.** The paper uses LAION-400M ViT-B/16+;
   `configs/methods/winclip.yaml` records ours as `unverified_until_first_colab_run`. Same
   architecture with different pre-training is a different frozen feature space.
3. **Determinism, measured not assumed.** One category twice at two seeds. Bit-identical image
   scores mean one run is reportable with that measurement recorded beside it; anything else means
   three seeds and three calibrations. The targets file already pre-registers this procedure.

### C — GPU: the two gates

MVTec AD classic (15 categories) at **91.8 ± 1.0** I-AUROC and VisA (12 objects) at **78.1 ± 1.0**,
both from the method's own paper at the configuration this repo runs (protocol §2 v0.2.14). Both
must pass before any MVTec AD 2 number for WinCLIP is reported. All 27 per-category values are in
`configs/reproduction/winclip.yaml`, so a miss localises to a category.

Both datasets download directly; neither needs Drive.

**A measured cost probe comes first.** PatchCore's cost model does not transfer: WinCLIP builds no
memory bank, so the quadratic coreset cost that made VisA four hours does not apply — but inference
over 27 categories with a 242-prompt ensemble per image has never been timed. Time one category,
then plan sessions. This is the same device SAA+'s plan pre-registers in place of an assumption.

**Table 7's VisA 78.9 is not the target.** It is the `+ specific states` ablation, which adds
per-object defect words §3 forbids us to write. It is higher than the real target and therefore the
tempting number; it sits in the targets file under `not_the_target`.

### D — GPU: WinCLIP's MVTec AD 2 grid

Eight categories, `test_public`, seeds per B's measurement, archives fetched in-session
(`docs/datasets-access.md`). `--max-bytes 12000000000` covers every category.

### E — GPU + CPU: validation scoring and calibration

`validation` was deliberately unscored during M3. PatchCore needs three seeds × eight categories;
WinCLIP one or three × eight. Then `calibrate_threshold.py` per seed per category —
`per_image_robust_z`, `alpha = 1e-3`, the rule designated for submission on 2026-08-20 — writing
`configs/thresholds/<dataset>__<method>.yaml`.

**Three calibrations per (method, category) for PatchCore**, all committed, submitting seed 0.
Decided 2026-10-01: the spread of `k` across seeds is a measurement nobody has published, and
committing all three makes it auditable that the submitted seed was not chosen for its score. The
tool already refuses pooled seeds and any split other than `validation`.

**The artifact is committed before any submission.** That commit is the pre-registration, and a
submission whose artifact was committed later is a violation detectable from the log.

### F — CPU: submission packaging, after first login

Confirm the metric definition, payload format and attempt limit against the server's own docs —
three unticked checkboxes in `docs/datasets-access.md`, currently filled from the VAND 3.0 report,
a secondary source. Then build the payload. One submission per method (§7, stricter than the
server's two per week).

### G — figures: regenerate maps and plot

Re-run `test_public` for the two or three illustrated categories so their anomaly maps exist again
(Walnuts and Can for figure 4, Sheet Metal for figure 2), then write the plotting code — CPU, no
model. **These are PatchCore re-runs, not WinCLIP's**, so what D and E provide is the archives on
the VM, not the maps themselves. Procedure, legitimacy check and the selection rule in section 5.

## 3. What is cut, and how it is recorded

Protocol amendment **v0.2.18**, four clauses:

1. **Methods.** The study benchmarks `patchcore_ref` and `winclip`. AnomalyCLIP, AdaCLIP, SAA+ and
   the Qwen2.5-VL baseline are **not benchmarked**, and their adapters, configs, overlap audits,
   prompt provenance and pre-registered gates **stay in the repository** — they are C3's evidence,
   and deleting them would destroy the record that makes C3 auditable. The four were not dropped
   for being uninteresting: benchmarking them is expensive in ways the paper documents, which is
   C3's claim, so the cut is evidence for the contribution rather than a concession against it.
2. **M5 out.** §5 stays as written, marked **not exercised in this study**. No latency numbers, no
   Table 3, and the planned accuracy-vs-latency scatter is dropped (the figures in section 5 are
   numbered independently of the original plan's); §7 narrows to what the accuracy results support.
   §6.5's cost paragraph **stays** — it reports measured accelerator time for *validation*, not
   inference latency, and the paper must not let the two be confused.
3. **Private split in.** `per_image_robust_z`, three calibrations per (method, category) for
   PatchCore, one submission per method. The `\todo{withdraw if the threshold rule is not built
   before submission}` tripwires on C1 and C3 come out once a real calibration has run — not on
   `intensity_baseline`, which is the floor and not a detector.
4. **What does not change.** §2's gates, §4's metrics, §6's seed rule, §7's prohibition on
   reconfiguring after seeing a result. **The amendment reduces the method list; it relaxes no
   evaluation rule.**

## 4. Verification

| item | how it is verified |
|---|---|
| A | TDD. `run_mvtec_ad2.py`'s ten existing tests stay green; new cases per method. The real risk is silently changing PatchCore's path, so a test asserts it constructs the same backend as before. |
| B | A staged probe, the pattern `notebooks/probe_patchcore.py` established — which is what found the double-normalisation bug that survived weeks of passing tests. Plus the three pre-registered checks before any scoring, plus a separation smoke test. |
| C | The gates *are* the verification: ±1.0 against 91.8 and 78.1, with per-category values so a miss localises. |
| D | The guards M3 exercised eight times: layout verification, per-seed shard copy, the memory guard, the report writer. The AU-PRO random baseline is the reference that says whether a WinCLIP number means anything. |
| E | `calibrate_threshold.py` already refuses pooled seeds and non-`validation` splits. The committed artifact before submission is the audit. |
| F | See the gap below. |
| G | Image scores of the regenerated run must match the committed shard to full precision. |

### The verification gap in F, and its mitigation

A private-split payload cannot be checked before sending: the server returns a score, and a
malformed payload can return a plausible wrong number. With one submission per method, an error
spends the only attempt.

**Mitigation, which costs nothing:** package the **public** split through the identical code path
and verify it reproduces our own `seg_f1_at` figures. A packaging defect then surfaces against
ground truth we hold, rather than against a number we cannot audit.

### Pre-registered contingency, recorded before the gate runs

If WinCLIP misses either gate by more than 1.0, §2 blocks every MVTec AD 2 number for it and the
paper loses C2. **In that case the paper falls back to C1 + C3 alone** — anchor and methodology,
no zero-shot numbers — with §1–§3 rewritten accordingly. Writing this now is what makes it a
contingency rather than a decision taken after seeing a failure.

## 5. Figures

The anomaly maps from M3 do not exist: they were deliberately discarded (24.9 GB across the grid,
against Drive's free 15 GB). Shards hold image scores and paths, not maps. Every qualitative figure
therefore needs maps **regenerated**.

That is nearly free. Regeneration is deterministic — demonstrated 2026-09-29, when Fruit Jelly
returned the same four values to four decimals nine days later on a different VM — and **M4 already
requires every archive back on the VM**, so figure runs ride along at no extra upload. The runner's
own missing-maps message documents the procedure: delete that seed's shard and re-run.

**Legitimacy check:** after regenerating, compare `image_score` against the committed shard at full
precision. If they match, the maps are *the ones that produced the reported numbers*, not new maps
that resemble them. Without that check a figure illustrates a different run from the table.

**This is why item A may not touch the scoring path.**

### The four figures

1. **What is in the dataset.** One scene across its lighting conditions (`regular`, `overexposed`,
   `underexposed`, `shift_*`), plus an anomalous example with its ground-truth mask. No model
   required. Makes visible why the benchmark exists.
2. **The registered mechanism, shown.** Sheet Metal's 4224×1056 frame beside its square 256×256
   resize: 16.5× horizontal compression against 4.1× vertical stops being arithmetic. One image and
   one resize.
3. **Pixels against pixels.** For one anomalous image: original, ground-truth mask, anomaly map as a
   heatmap, and the map thresholded at the pre-registered threshold. This is what is compared to
   what.
4. **Working against failing**, same layout, two columns: **Walnuts**, where the anchor localises at
   18× the random baseline, against **Can**, where we measured the map to be *dimmer inside* the
   ground-truth region than outside on five of eight sampled anomalous images (ratios 0.796–0.880).
   That panel shows *how* it fails, not that it fails.

A fifth, quantitative and free of maps: per-condition metrics as small multiples, straight from the
shards.

### The anti-cherry-picking rule, pre-registered before any map is viewed

A hand-picked example can show anything. The rule: **the median-scoring anomalous image** of that
condition and seed, declared in advance, applied, with the seed and condition in the caption. If
the median turns out to be unilluminating, it is reported as such; no second search.

**Cost:** two or three categories regenerated (Walnuts, Can, Sheet Metal), minutes of fitting each,
over archives M4 needs anyway. Plus plotting code, which is CPU-only.

## Dependencies

```
A (CPU) ──┬── B (GPU) ── C (gates) ── D (grid) ──┬── E (calibration) ── F (submission)
          │                                       │
          └───────────────────────────────────────┴── G (figures, needs archives from D/E)
```

A is the only item that can start now. C gates D by protocol §2. G needs the archives D and E
already require, but its runs are PatchCore's `test_public` re-runs rather than any WinCLIP output,
so G could in principle run before B if an archive is on a VM for another reason.

## Out of scope, explicitly

No further methods. No M5. No aspect-ratio experiment on Sheet Metal — §7 permits it as a separate
dated experiment, but it supports none of the three contributions and would extend scope.
