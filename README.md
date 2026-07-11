# vlm-anomaly-bench

**Zero-shot & vision-language anomaly detection, evaluated where nobody has looked yet: MVTec AD 2 and Real-IAD Variety.**

> Our [survey of industrial visual anomaly detection (2020–2026)](https://doi.org/10.5281/zenodo.XXXXXXX) found that
> top methods report >99% I-AUROC on MVTec AD — and that **not a single surveyed method reports results on the
> 2025 frontier benchmarks** MVTec AD 2 and Real-IAD Variety. This repository closes that gap for the
> zero-shot / VLM family, with a reproducible protocol and honest, efficiency-aware metrics.

**Status:** scaffolding — protocol frozen before any results are computed (see [`docs/protocol.md`](docs/protocol.md)).

## Why this benchmark

- **Benchmark saturation is not method convergence.** The same methods drop by up to 12.9 I-AUROC points
  when moved from MVTec AD to Real-IAD. Frontier benchmarks exist precisely to restore discrimination —
  but the zero-shot/VLM literature hasn't touched them yet.
- **Zero-shot matters industrially.** Rare defects and short product cycles mean training data often
  doesn't exist. If VLM-based methods are the answer, they must prove it on hard, current benchmarks.
- **Post-saturation metrics.** We report localisation quality (AU-PRO / SegF1) and deployment cost
  (latency, params, VRAM) alongside AUROC — the axes that still separate methods in practice.

## Methods under evaluation

| Method | Family | Shots | Source |
|---|---|---|---|
| WinCLIP / WinCLIP+ | CLIP windows, handcrafted prompts | 0 / few | CVPR 2023 |
| AnomalyCLIP | learned object-agnostic prompts | 0 (aux-trained) | ICLR 2024 |
| AdaCLIP | hybrid learnable prompts | 0 (aux-trained) | ECCV 2024 |
| SAA+ | GroundingDINO + SAM cascade | 0 | 2023 |
| MLLM baseline (Qwen2.5-VL) | generalist multimodal LLM, structured prompting | 0 | 2025 |
| PatchCore *(reference anchor)* | memory bank (full-shot) | full | CVPR 2022 |

The full-shot PatchCore anchor calibrates every table: it answers "how far is zero-shot from the
classical ceiling *on the same frontier data*?"

## Datasets

| Dataset | Role | Access |
|---|---|---|
| MVTec AD 2 | primary frontier benchmark (lighting shifts) | public download, research license — see [`docs/datasets-access.md`](docs/datasets-access.md) |
| Real-IAD Variety | primary frontier benchmark (scale/variety) | requires application — start early |
| VisA | sanity check vs published numbers | public |
| MVTec AD (classic) | calibration vs literature | public |

Raw data is **never** committed. `scripts/prepare_data.py` verifies checksums and folder layout.

## Metrics

Image level: I-AUROC, I-AP, I-F1max · Pixel level: P-AUROC, AU-PRO (FPR≤0.3 and ≤0.05), SegF1
(MVTec AD 2 official protocol) · Efficiency: median latency (ms/image, fixed hardware), parameters,
peak VRAM. Every number ships with the config hash + seed that produced it.

## Repository layout

```
configs/            # one YAML per dataset and per method — single source of truth for every run
src/vlmab/
  datasets/         # loaders with a common sample schema (image, label, mask, meta)
  methods/          # one adapter per method implementing AnomalyMethod.predict()
  metrics/          # image/pixel/efficiency metrics, unit-tested against toy cases
  eval/             # runner: dataset x method -> results parquet
  viz/              # tables (markdown/LaTeX) and figures from results
scripts/            # prepare_data, run_eval, make_tables — the full pipeline as CLI
tests/              # metric correctness + adapter contract tests (no downloads needed)
docs/               # frozen protocol, dataset access notes
results/            # generated tables/figures (source parquet tracked via releases)
```

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/prepare_data.py --check          # verifies dataset layout
.venv/bin/python scripts/run_eval.py --method winclip --dataset mvtec_ad2
.venv/bin/python scripts/make_tables.py                   # regenerates results/tables
```

## Roadmap

- [ ] M1 — protocol frozen + dataset access secured (Real-IAD application sent)
- [ ] M2 — metrics module tested; PatchCore anchor + WinCLIP on VisA reproduce published numbers (±1pt)
- [ ] M3 — full grid on MVTec AD 2
- [ ] M4 — full grid on Real-IAD Variety
- [ ] M5 — efficiency pass on fixed hardware; tables + figures frozen
- [ ] M6 — preprint on arXiv/Zenodo; submission (target: CVPR VAND workshop / EAAI)

## Companion paper

The manuscript lives in its own repository: **vlm-anomaly-paper** (plan and venue strategy in its
`OUTLINE.md`). Tables and figures are generated *here* (`scripts/make_tables.py`) and pulled into
the paper — numbers are never hand-edited.

## Author

**Sergio Duarte** — Machine Vision & AI Engineer (MSc, University of Luxembourg).
Companion to the survey *Post-Saturation Industrial Visual Anomaly Detection* (Zenodo, 2026).

## License

MIT for the code in this repository. Datasets keep their own licenses; nothing from them is redistributed here.
