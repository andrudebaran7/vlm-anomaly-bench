#!/usr/bin/env python3
"""Experiment A: score one MVTec AD 2 category with the forward resize aliased and antialiased,
on one VM, same seeds, and compare both against the committed reports.

Why (docs/next-steps.md, "CHECKED 2026-10-30"): the pre-v0.2.21 grid ran on Colab torch
2.11.0+cu128 and the v0.2.21 grid on cu130, so the before/after differs in the resize AND the
CUDA build. Only both configurations on one runtime separate the two. Two readings, each a check:

* `antialiased` here should reproduce the committed v0.2.21 report (same build, same code) --
  if it does, this VM is the one the reported numbers came from;
* `aliased` here against the committed pre-v0.2.21 report (cu128) measures the build's effect
  with the resize held fixed. If it matches, the build moved nothing and the before/after is the
  resize alone.

    python scripts/antialias_ablation.py --category sheet_metal --max-bytes 12000000000

Writes to its own root (`results/antialias_ablation/<category>/<mode>/`), never to Drive and never
under a registered label: the aliased configuration is superseded and is a diagnostic only.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_mvtec_ad2 import (DRIVE_ARCHIVES, PREPROCESS, SUMMARY_METRICS,  # noqa: E402
                           fetch, verify_layout)

#: mode -> the backend's `diagnostic_aliased_forward`.
MODES = {"aliased": True, "antialiased": False}
#: The committed report each mode is compared against.
REPORTS = {"aliased": Path("results/mvtec_ad2/patchcore_ref"),
           "antialiased": Path("results/mvtec_ad2/patchcore_ref_v0.2.21")}


def seed_stats(per_seed: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean and std over seeds per lighting condition, as `run_mvtec_ad2.summarise` does."""
    combined = pd.concat(per_seed, ignore_index=True)
    metrics = [m for m in SUMMARY_METRICS if m in combined.columns]
    return combined.groupby("meta_lighting")[metrics].agg(["mean", "std"])


def compare(stats: pd.DataFrame, report_text: str) -> list[dict]:
    """Each measured seed mean against the report's, at the report's own decimals."""
    from vlmab.viz.tables import parse_report

    reported = {r.condition: r for r in parse_report(report_text)}
    rows = []
    for condition, row in stats.iterrows():
        if condition not in reported:
            raise KeyError(f"condition {condition!r} is not in the report ({sorted(reported)})")
        for metric in SUMMARY_METRICS:
            if (metric, "mean") not in row.index:
                continue
            want = str(reported[condition].mean[metric])
            places = len(want.split(".")[1])
            got = f"{row[(metric, 'mean')]:.{places}f}"
            rows.append({"condition": condition, "metric": metric, "measured": got,
                         "reported": want, "equal": got == want})
    return rows


def check_mode(df: pd.DataFrame, mode: str) -> None:
    """The shards must record the configuration that ran, or the comparison means nothing."""
    if "resize_antialias" not in df.columns:
        raise RuntimeError("the shards carry no resize_antialias column; cannot tell the modes apart")
    want = not MODES[mode]
    found = set(df["resize_antialias"].dropna().astype(bool))
    if found != {want}:
        raise RuntimeError(f"{mode} shards record resize_antialias={sorted(found)}, not [{want}]")


def run_mode(category: str, root: Path, out: Path, mode: str, seeds: Sequence[int],
             device: str) -> None:  # pragma: no cover - GPU
    from vlmab.datasets.mvtec_ad2 import MVTecAD2
    from vlmab.eval.provenance import method_meta
    from vlmab.eval.runner import run_evaluation
    from vlmab.eval.store import ResultStore
    from vlmab.methods.patchcore_backend import PatchCoreBackend
    from vlmab.methods.patchcore_ref import PatchCoreRef

    store = ResultStore(out / "shards")
    for seed in seeds:
        # A fresh backend per seed, as every runner: a second fit on one backend is a no-op.
        method = PatchCoreRef(backend=PatchCoreBackend(
            seed=seed, preprocess=PREPROCESS, diagnostic_aliased_forward=MODES[mode]))
        meta = method_meta(method, {"method": "patchcore_ref", "split": "test_public",
                                    "experiment": "antialias_ablation"})
        run_evaluation(MVTecAD2(root), method, store, meta, categories=[category],
                       split="test_public", fit_split="train", maps_dir=out / "maps",
                       device=device)
        print(f"  {mode} seed {seed}: done (resize_antialias {method.resize_antialias})")


def summarise_mode(out: Path, mode: str, max_bytes: int | None) -> pd.DataFrame:  # pragma: no cover
    from vlmab.eval.aggregate import aggregate
    from vlmab.eval.store import ResultStore

    df = ResultStore(out / "shards").load_all()
    check_mode(df, mode)
    kw = {} if max_bytes is None else {"max_bytes": max_bytes}
    per_seed = [aggregate(g, by="meta_lighting", **kw).assign(seed=s)
                for s, g in df.groupby("seed")]
    return seed_stats(per_seed)


def print_mode(category: str, mode: str, stats: pd.DataFrame) -> list[dict]:
    report = REPORTS[mode] / f"{category}.md"
    print(f"\n== {mode} (this VM) vs the committed {report}")
    print(f"  {'lighting':<14}{'metric':<12}{'this VM':>10}{'report':>10}  same")
    rows = compare(stats, report.read_text())
    for r in rows:
        print(f"  {r['condition']:<14}{r['metric']:<12}{r['measured']:>10}{r['reported']:>10}  "
              f"{'yes' if r['equal'] else 'NO'}")
    same = sum(r["equal"] for r in rows)
    print(f"  {same} of {len(rows)} cells equal at the report's decimals")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--category", default="sheet_metal")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--root", type=Path, default=Path("data/mvtec_ad2"))
    parser.add_argument("--archives", type=Path, default=DRIVE_ARCHIVES)
    parser.add_argument("--results", type=Path, default=Path("results/antialias_ablation"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-bytes", type=int, default=None)
    args = parser.parse_args(argv)

    head = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse",
                           "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"{Path(__file__).name} running from commit {head or 'unknown'}")
    try:
        import torch
        print(f"torch {torch.__version__}, CUDA available {torch.cuda.is_available()}\n")
    except ImportError:
        print("torch not importable\n")

    fetch(args.category, args.root, args.archives)
    verify_layout(args.category, args.root)

    stats = {}
    for mode in MODES:
        out = args.results / args.category / mode
        print(f"[{mode}] {args.category}, seeds {args.seeds}")
        run_mode(args.category, args.root, out, mode, args.seeds, args.device)
        stats[mode] = summarise_mode(out, mode, args.max_bytes)
        print_mode(args.category, mode, stats[mode])

    print(f"\n== I-AUROC per condition, seed mean (this VM): aliased -> antialiased")
    for condition in stats["antialiased"].index:
        a = stats["aliased"].loc[condition, ("i_auroc", "mean")]
        b = stats["antialiased"].loc[condition, ("i_auroc", "mean")]
        print(f"  {condition:<14}{a:.4f} -> {b:.4f}  ({b - a:+.4f})")
    means = {m: stats[m][("i_auroc", "mean")].mean() for m in MODES}
    print(f"  {'mean':<14}{means['aliased']:.4f} -> {means['antialiased']:.4f}  "
          f"({means['antialiased'] - means['aliased']:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
