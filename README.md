# vlm-anomaly-bench

**What a passing reproduction check does not certify: a gated PatchCore reference on MVTec AD 2.**

A reproduction check, matching a method's published score on its original benchmark, is the usual
safeguard before a pipeline is trusted on a new one. This repository documents a case in which the
check passes and still misses what decides the verdict on the new dataset. Our full-shot PatchCore
pipeline built its memory bank from antialiased images and scored test images resized without
antialiasing. It passed its MVTec AD check (98.02 ± 0.07 I-AUROC against a published 99.0), yet
left Sheet Metal at chance on MVTec AD 2 (0.510). A same-runtime ablation with the same seeds
attributes the change to the resize alone. The corrected pipeline is reported as a reference on all
eight categories of the public split. The study was planned as a zero-shot comparison against that
reference; the one zero-shot method carried to its reproduction check, WinCLIP, did not pass it, so
no MVTec AD 2 number is reported for it. Protocol, code and every reported number are here.

**Status (2026-11-03):** protocol at v0.2.30 ([`docs/protocol.md`](docs/protocol.md)); MVTec AD 2
public-split grid complete for the anchor; same-runtime ablation run on Sheet Metal, Fabric and the MVTec AD gate; test-set intervals for I-AUROC;
private split and evaluation-server submission not run. The paper is in preparation. Operational
state and history: [`docs/next-steps.md`](docs/next-steps.md).

## Results

**Reproduction checks**, MVTec AD classic, I-AUROC, tolerance ±1.0 on the mean over 15 categories:

| Method | Measured | Published | Verdict | Report |
|---|---|---|---|---|
| PatchCore (anchor) | 98.22 ± 0.03 (3 seeds) | 99.0 | pass | [`patchcore_mvtec_ad_v0.2.21.md`](results/reproduction/patchcore_mvtec_ad_v0.2.21.md) |
| WinCLIP | 90.37 (deterministic, 1 run) | 91.8 | **fail** | [`winclip_mvtec_ad.md`](results/reproduction/winclip_mvtec_ad.md) |

Under the protocol (§2, v0.2.24), a method that fails its gate gets no MVTec AD 2 number: its gate
report is published and its MVTec AD 2 results are withheld. That is what happened to WinCLIP.

**The PatchCore anchor on MVTec AD 2** (`test_public`, three seeds, mean over lighting conditions of
the per-condition mean over seeds; AU-PRO also as a multiple of its random baseline):

| Category | I-AUROC | AU-PRO@0.3 | AU-PRO@0.05 | Usable reference |
|---|---|---|---|---|
| Vial | 0.888 | 0.863 (5.75×) | 0.567 (22.70×) | yes |
| Walnuts | 0.861 | 0.746 (4.97×) | 0.422 (16.89×) | yes |
| Sheet Metal | 0.797 | 0.228 (1.52×) | 0.068 (2.72×) | yes |
| Fruit Jelly | 0.773 | 0.634 (4.23×) | 0.379 (15.15×) | yes |
| Fabric | 0.723 | 0.254 (1.69×) | 0.028 (1.12×) | yes |
| Rice | 0.537 | 0.433 (2.89×) | 0.173 (6.91×) | no |
| Wall Plugs | 0.485 | 0.351 (2.34×) | 0.114 (4.55×) | no |
| Can | 0.452 | 0.237 (1.58×) | 0.015 (0.61×) | no |

The random baseline is half the false-positive limit: 0.15 at 0.3 and 0.025 at 0.05. Mean over all
eight categories: I-AUROC 0.689, AU-PRO@0.3 0.468, AU-PRO@0.05 0.221; over the five usable ones:
0.808, 0.545, 0.293.

Per-condition and per-seed values are in
[`results/mvtec_ad2/patchcore_ref_v0.2.21/`](results/mvtec_ad2/patchcore_ref_v0.2.21/). The
usable/unusable criterion is on detection only and was fixed **after** these numbers existed
(protocol v0.2.22); it is stated as post hoc wherever it is used.

**What the reproduction check did not see.** Until protocol v0.2.21 the anchor's memory bank was
built from antialiased resizes and its test queries were scored without antialiasing. That
configuration passed the MVTec AD check at 98.02 ± 0.07. On MVTec AD 2, correcting it changed the
verdict on two categories of this test set, and none of the other six moved by more than 0.064
I-AUROC. Under resampling of test scenes (protocol §6 v0.2.26;
[`results/bootstrap/iauroc_bootstrap.md`](results/bootstrap/iauroc_bootstrap.md)) the paired change
excludes zero on Sheet Metal (+0.287, 95% interval [+0.086, +0.504]) and not on Fabric (+0.160,
[-0.096, +0.412]). These intervals can be recomputed from the published image-level scores in
[`results/image_scores/`](results/image_scores/README.md) (CC BY-NC-SA 4.0):

| Category | Configuration | I-AUROC | AU-PRO@0.3 | AU-PRO@0.05 |
|---|---|---|---|---|
| Sheet Metal | aliased (first run) | 0.510 | 0.252 (1.68×) | 0.068 (2.71×) |
| Sheet Metal | antialiased (corrected) | 0.797 | 0.228 (1.52×) | 0.068 (2.72×) |
| Fabric | aliased (first run) | 0.562 | 0.146 (0.97×) | 0.008 (0.30×) |
| Fabric | antialiased (corrected) | 0.723 | 0.254 (1.69×) | 0.028 (1.12×) |

Both configurations of each category were re-run on one runtime with the same three seeds, and
each reproduced its committed grid in all 18 cells to four decimals, so the change is the
pre-processing alone. This ablation was added after the results existed, as a diagnostic (protocol
v0.2.23); the runs are recorded in [`docs/next-steps.md`](docs/next-steps.md), "RESULT 2026-10-30 —
experiment A on Sheet Metal" and "RESULT 2026-10-31 — experiment A on Fabric". The check itself
moved from 98.02 to 98.22, and the same re-run on MVTec AD (protocol v0.2.25; "RESULT 2026-11-03 —
experiment B") shows that is the resize alone too: each configuration reproduced its committed
report in all 19 printed values, to the reports' two decimals. The superseded numbers are kept, marked, in
[`results/mvtec_ad2/patchcore_ref/`](results/mvtec_ad2/patchcore_ref/) and the older reports under
`results/reproduction/`; none of them is an anchor number. The table is
`results/tables/ablation_rows.tex`. The pass itself has a history: a first single-seed run missed the
check at 97.94, and the rule that the verdict is on the mean of the seed means (protocol §6, v0.2.13) was
written after that miss and before the three-seed run that passed.

The mismatch is known upstream, and we do not claim its discovery: anomalib's export path drops the
antialiasing its training and test loops apply ([#2944](https://github.com/open-edge-platform/anomalib/issues/2944),
[#3726](https://github.com/open-edge-platform/anomalib/issues/3726), proposed fix
[#3882](https://github.com/open-edge-platform/anomalib/pull/3882)). What this repository adds is what it does to a
reproduction check and to a target benchmark.

**Against other PatchCore results on MVTec AD 2** (protocol v0.2.28, post hoc, a sanity check and not a gate;
[`results/comparison/patchcore_literature.md`](results/comparison/patchcore_literature.md)): the category order agrees with
the dataset authors' at both AU-PRO limits (Spearman 0.857 to 0.976, other split and backbones). It agrees less with Vo et
al.'s PatchCore-style baseline on I-AUROC (0.619), which at 448×448 leaves Sheet Metal at 0.558, so the anchor's
usable/not-usable split belongs to this pipeline and not to the categories.

## Scope

Protocol v0.2.18 narrowed the study to two methods run end to end, PatchCore (full-shot anchor)
and WinCLIP (zero-shot). AnomalyCLIP, AdaCLIP, SAA+ and a Qwen2.5-VL-3B baseline have CPU-tested
adapters here but were not benchmarked; what reading their papers and released artefacts showed
is part of the paper's analysis. Efficiency measurements (latency, VRAM) are out of scope.

| Dataset | Role |
|---|---|
| MVTec AD 2 | the target benchmark (eight categories, lighting conditions); see [`docs/datasets-access.md`](docs/datasets-access.md) |
| MVTec AD (classic) | per-method reproduction check against published numbers |
| VisA | PatchCore: secondary check, reported, never gating (86.26 against a third-party 92.4, earlier configuration, [`patchcore_ref_visa.md`](results/reproduction/patchcore_ref_visa.md)). WinCLIP: a gate under the protocol, not run because its MVTec AD gate had already failed |

Raw data is never committed.

**Metrics reported:** I-AUROC and AU-PRO at false-positive limits 0.3 and 0.05, with AU-PRO's
random baseline (half the limit for an uninformative map; pinned by
`tests/test_au_pro_random_baseline.py`). SegF1 needs a ground-truth-free threshold; the rule is
built (`scripts/calibrate_threshold.py`) but no calibration has been run.

## Reproduce

Tests (no dataset, no GPU):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/python -m pytest -q
```

GPU runs were made on Google Colab (Tesla T4) with anomalib 2.6.0 and torch 2.11.0+cu130, driven
by the notebooks in [`notebooks/`](notebooks/). Those versions are pinned in
[`requirements-gpu.txt`](requirements-gpu.txt) (also `pip install -e .[gpu]`);
`requirements.txt` is the CPU core the tests need. Each report records the commit, seed, GPU and library versions that produced it.

- PatchCore anchor: `notebooks/patchcore_colab.ipynb`, phase 6, which runs
  `scripts/run_mvtec_ad2.py --category <c>` per category.
- Reproduction checks: `scripts/run_mvtec_ad.py --method <m>` followed by
  `scripts/reproduction_gate.py`; WinCLIP's is cells 4.1–4.4 of `notebooks/winclip_colab.ipynb`.

The anchor tables are generated from the committed reports: `scripts/make_tables.py` writes
`results/tables/anchor_rows.tex` (the anchor table above), `anchor_per_condition.tex` (every
lighting condition's mean ± std over seeds) and `ablation_rows.tex` (the before/after table), and
prints both means; and `scripts/make_figures.py` (needs `pip install -e .[plots]`)
draws `results/figures/anchor_per_condition.pdf` from the same reports.

## Repository layout

```
configs/    one YAML per dataset, per method and per reproduction target
src/vlmab/  datasets, method adapters, metrics, resumable runner, threshold rules
scripts/    run_mvtec_ad2, run_mvtec_ad, reproduction_gate, calibrate_threshold, prepare_data, ...
notebooks/  the Colab drivers for the GPU runs
tests/      metric correctness, adapter contracts, runner and provenance (no downloads needed)
docs/       protocol (versioned, with changelog), dataset access, operational record
results/    committed reports and image-level scores; shards and anomaly maps stay on Drive
```

## Author

**Sergio Duarte** — Independent Researcher, Luxembourg · saduartep@gmail.com ·
ORCID [0009-0009-7079-0160](https://orcid.org/0009-0009-7079-0160).
Companion to the survey *Post-Saturation Industrial Visual Anomaly Detection: A Cross-Vertical
Comparative Survey (2020–2026)* (Zenodo, 2026, [doi:10.5281/zenodo.20649845](https://doi.org/10.5281/zenodo.20649845)).

## License

MIT for the code in this repository. Datasets keep their own licenses; nothing from them is
redistributed here.
 The exception is `results/image_scores/`, scores derived from MVTec AD 2 and released under its CC BY-NC-SA 4.0; see the [README there](results/image_scores/README.md).