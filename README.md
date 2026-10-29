# vlm-anomaly-bench

**A full-shot anchor and a reproduction record for zero-shot anomaly detection on MVTec AD 2.**

This repository set out to evaluate zero-shot and vision-language anomaly detection on MVTec AD 2,
the 2025 benchmark on which no zero-shot or VLM-based method appears among the published baselines
or the official leaderboard. It reports instead why that comparison could not be made as planned:
the full-shot PatchCore anchor is a usable detection ceiling on only five of the eight categories,
and the one zero-shot method carried to its reproduction check, WinCLIP, did not pass it. Protocol,
code and every reported number are here.

**Status (2026-10-29):** protocol at v0.2.22 ([`docs/protocol.md`](docs/protocol.md)); MVTec AD 2
public-split grid complete for the anchor; private split and evaluation-server submission not run.
Operational state and history: [`docs/next-steps.md`](docs/next-steps.md).

## Results

**Reproduction checks**, MVTec AD classic, I-AUROC, tolerance ±1.0 on the mean over 15 categories:

| Method | Measured | Published | Verdict | Report |
|---|---|---|---|---|
| PatchCore (anchor) | 98.22 ± 0.03 (3 seeds) | 99.0 | pass | [`patchcore_mvtec_ad_v0.2.21.md`](results/reproduction/patchcore_mvtec_ad_v0.2.21.md) |
| WinCLIP | 90.37 (deterministic, 1 run) | 91.8 | **fail** | [`winclip_mvtec_ad.md`](results/reproduction/winclip_mvtec_ad.md) |

WinCLIP's failure means, under the protocol, that no MVTec AD 2 number is reported for it.

**The PatchCore anchor on MVTec AD 2** (`test_public`, three seeds, mean over lighting conditions of
the per-condition mean over seeds; AU-PRO also as a multiple of its random baseline):

| Category | I-AUROC | AU-PRO@0.3 | AU-PRO@0.05 | Usable ceiling |
|---|---|---|---|---|
| Vial | 0.888 | 0.863 (5.81×) | 0.567 (22.79×) | yes |
| Walnuts | 0.861 | 0.746 (5.02×) | 0.422 (16.96×) | yes |
| Sheet Metal | 0.797 | 0.228 (1.53×) | 0.068 (2.73×) | yes |
| Fruit Jelly | 0.773 | 0.634 (4.26×) | 0.379 (15.21×) | yes |
| Fabric | 0.723 | 0.254 (1.71×) | 0.028 (1.12×) | yes |
| Rice | 0.537 | 0.433 (2.91×) | 0.173 (6.94×) | no |
| Wall Plugs | 0.485 | 0.351 (2.36×) | 0.114 (4.57×) | no |
| Can | 0.452 | 0.237 (1.59×) | 0.015 (0.62×) | no |

Per-condition and per-seed values are in
[`results/mvtec_ad2/patchcore_ref_v0.2.21/`](results/mvtec_ad2/patchcore_ref_v0.2.21/). The
usable/unusable criterion is on detection only and was fixed **after** these numbers existed
(protocol v0.2.22); it is stated as post hoc wherever it is used.

**Superseded configuration, kept as evidence.** Until protocol v0.2.21 the anchor's memory bank was
built from antialiased resizes and its test queries were scored without antialiasing. Those
numbers are kept, marked, in [`results/mvtec_ad2/patchcore_ref/`](results/mvtec_ad2/patchcore_ref/)
and the older reports under `results/reproduction/`; none of them is an anchor number.

## Scope

Protocol v0.2.18 narrowed the study to two methods run end to end, PatchCore (full-shot anchor)
and WinCLIP (zero-shot). AnomalyCLIP, AdaCLIP, SAA+ and a Qwen2.5-VL-3B baseline have CPU-tested
adapters here but were not benchmarked; what reading their papers and released artefacts showed
is part of the paper's analysis. Efficiency measurements (latency, VRAM) are out of scope.

| Dataset | Role |
|---|---|
| MVTec AD 2 | the frontier benchmark (eight categories, lighting conditions) — see [`docs/datasets-access.md`](docs/datasets-access.md) |
| MVTec AD (classic) | per-method reproduction check against published numbers |
| VisA | secondary check, reported only, never gating |

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
by the notebooks in [`notebooks/`](notebooks/). The notebooks pin anomalib; torch is the build
Colab ships, and `requirements.txt` pins neither. Each report records the commit, seed, GPU and library versions that produced it.

- PatchCore anchor: `notebooks/patchcore_colab.ipynb`, phase 6, which runs
  `scripts/run_mvtec_ad2.py --category <c>` per category.
- Reproduction checks: `scripts/run_mvtec_ad.py --method <m>` followed by
  `scripts/reproduction_gate.py`; WinCLIP's is cells 4.1–4.4 of `notebooks/winclip_colab.ipynb`.

The paper's tables are transcribed from the committed reports and checked against them cell by
cell; `scripts/make_tables.py` is not yet implemented.

## Repository layout

```
configs/    one YAML per dataset, per method and per reproduction target
src/vlmab/  datasets, method adapters, metrics, resumable runner, threshold rules
scripts/    run_mvtec_ad2, run_mvtec_ad, reproduction_gate, calibrate_threshold, prepare_data, ...
notebooks/  the Colab drivers for the GPU runs
tests/      metric correctness, adapter contracts, runner and provenance (no downloads needed)
docs/       protocol (versioned, with changelog), dataset access, operational record
results/    committed reports; shards and anomaly maps stay on Drive
```

## Author

**Sergio Duarte** — Machine Vision & AI Engineer (MSc, University of Luxembourg).
Companion to the survey *Post-Saturation Industrial Visual Anomaly Detection: A Cross-Vertical
Comparative Survey (2020–2026)* (Zenodo, 2026; DOI pending).

## License

MIT for the code in this repository. Datasets keep their own licenses; nothing from them is
redistributed here.
