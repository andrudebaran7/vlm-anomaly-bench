# vlm-anomaly-bench

**Zero-shot & vision-language anomaly detection, evaluated where nobody has looked yet: MVTec AD 2.**

> Our [survey of industrial visual anomaly detection (2020–2026)](https://doi.org/10.5281/zenodo.XXXXXXX) found that
> top methods report >99% I-AUROC on MVTec AD — and that **not a single surveyed method reports results on the
> 2025 frontier benchmark** MVTec AD 2. This repository closes that gap for the
> zero-shot / VLM family, with a reproducible protocol and honest, efficiency-aware metrics.

**Status:** protocol frozen at v0.2.13 (see [`docs/protocol.md`](docs/protocol.md)); the evaluation
core — metrics, resumable runner, MVTec AD 2 loader and lighting-grouped aggregation — is built and
tested (427 of 427 tests passing, no dataset or GPU required), verified end-to-end
against the real MVTec AD 2 layout. The CPU-testable half of every method adapter is done; only the
Colab GPU backends remain before the first results. No results computed yet.

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
| MLLM baseline (Qwen2.5-VL-3B) | generalist multimodal LLM, structured prompting | 0 | 2025 |
| PatchCore *(reference anchor)* | memory bank (full-shot) | full | CVPR 2022 |

The full-shot PatchCore anchor calibrates every table: it answers "how far is zero-shot from the
classical ceiling *on the same frontier data*?" The MLLM baseline runs the 3B variant: 7B in fp16
does not fit the 16 GB evaluation budget. Conclusions about MLLM capability are scoped to a 3B
generalist.

## Datasets

| Dataset | Role | Access |
|---|---|---|
| MVTec AD 2 | primary frontier benchmark (lighting shifts) | public download, research license — see [`docs/datasets-access.md`](docs/datasets-access.md) |
| VisA | sanity check vs published numbers | public |
| MVTec AD (classic) | calibration vs literature | public |

Real-IAD Variety is out of scope on compute grounds — see [docs/datasets-access.md](docs/datasets-access.md).

Raw data is **never** committed. `scripts/prepare_data.py` verifies a downloaded category's folder
layout, its per-lighting-condition image counts, and the integrity of every PNG before an
expensive run touches it.

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

# Working today:
.venv/bin/python -m pytest -q                                  # 304 tests, no dataset needed
.venv/bin/python scripts/prepare_data.py --root data/mvtec_ad2 # verify a downloaded category

# Once the Colab GPU backends land (M2):
.venv/bin/python scripts/run_eval.py --method winclip --dataset mvtec_ad2
.venv/bin/python scripts/make_tables.py                        # regenerates results/tables
```

## Roadmap

- [x] M1 — protocol frozen (v0.2.9); MVTec AD 2 downloaded and its layout, split names and 30.4 GB
  size verified from the real archive. Official metric names remain pending behind the
  evaluation-server registration.
- [ ] M2 — **core + adapter CPU-halves done**: pixel metrics, loader, resumable runner,
  lighting-grouped aggregation, and the CPU-testable half of seven adapters (intensity_baseline,
  mllm_qwen, patchcore_ref, winclip, anomalyclip, adaclip, saa), all tested. Remaining: the GPU
  backends and the ±1pt VisA reproduction, on Colab. **Operational state and ordering: [`docs/next-steps.md`](docs/next-steps.md).**
- [ ] M3 — full grid on the MVTec AD 2 public test split
- [ ] M4 — one evaluation-server submission per method; leaderboard numbers recorded *(registration
  requested 2026-07-23; see [`docs/datasets-access.md`](docs/datasets-access.md))*
- [ ] M5 — efficiency pass on the rented fixed instance; tables + figures frozen
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
