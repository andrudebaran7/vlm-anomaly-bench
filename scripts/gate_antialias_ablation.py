#!/usr/bin/env python3
"""Experiment B: the MVTec AD reproduction gate with the forward resize aliased and antialiased,
on one VM, same seeds, and both against the committed gate reports.

Why (docs/next-steps.md, "CHECKED 2026-10-30"; protocol v0.2.25): the gate passed at 98.02 ± 0.07
with the superseded aliased forward on Colab torch 2.11.0+cu128, and at 98.22 ± 0.03 with the
corrected antialiased forward on cu130. Those 0.20 points moved the resize AND the CUDA build.
Experiment A separated the two on Sheet Metal and Fabric; this does the same for the gate.
Three readings:

* `antialiased` here against the committed v0.2.21 gate report -- equal means this VM computes
  what the reported gate computed;
* `aliased` here against the committed pre-v0.2.21 gate report (cu128) -- equal means the build
  moved nothing;
* `aliased` against `antialiased`, both from this VM -- the correction's effect on the gate with
  the build held fixed. This one stands even if the first two do not match.

    python scripts/gate_antialias_ablation.py --mode aliased --seeds 0     # one (mode, seed)
    python scripts/gate_antialias_ablation.py --compare-only               # score and compare

Writes to its own root (`results/antialias_ablation/mvtec_ad/<mode>/`), never under a registered
label: the aliased configuration is superseded, and no number here is a result of the anchor.
Scoring is `reproduction_gate.py` itself, so both modes are scored exactly as the reports were.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from antialias_ablation import MODES, check_mode  # noqa: E402
from run_mvtec_ad2 import PREPROCESS  # noqa: E402

#: The committed gate report each mode is compared against.
REPORTS = {"aliased": Path("results/reproduction/patchcore_mvtec_ad_3seed.md"),
           "antialiased": Path("results/reproduction/patchcore_mvtec_ad_v0.2.21.md")}
TARGETS = Path("configs/reproduction/patchcore_ref.yaml")
RESULTS = Path("results/antialias_ablation/mvtec_ad")


def parse_gate_report(text: str) -> dict:
    """The numbers a gate report prints, as the strings it prints them: the headline mean ± std,
    each seed's mean, and each category's measured mean ± std. Strings, so a comparison is at the
    report's own two decimals and never at a float's."""
    head = re.search(r"Measured mean I-AUROC \*\*([\d.]+) ± ([\d.]+)\*\*", text)
    if head is None:
        raise ValueError("no 'Measured mean I-AUROC **m ± s**' line: not a multi-seed gate report")
    seeds = dict(re.findall(r"^\| (seed\d+) \| ([\d.]+) \|$", text, flags=re.M))
    categories = {c: m for c, m in
                  re.findall(r"^\| ([a-z_]+) \| ([\d.]+ ± [\d.]+) \| [\d.]+ \|", text, flags=re.M)}
    if not seeds or not categories:
        raise ValueError("the report has no per-seed or no per-category table")
    return {"mean": head.group(1), "std": head.group(2), "seeds": seeds,
            "categories": categories}


def compare_reports(measured: dict, reported: dict) -> list[dict]:
    """Every printed number of the run on this VM against the committed report's."""
    rows = [{"what": "gate mean", "measured": f"{measured['mean']} ± {measured['std']}",
             "reported": f"{reported['mean']} ± {reported['std']}"}]
    for key in ("seeds", "categories"):
        missing = sorted(set(reported[key]) - set(measured[key]))
        if missing:
            raise KeyError(f"this VM's report lacks {key} {missing}")
        rows += [{"what": name, "measured": measured[key][name], "reported": value}
                 for name, value in reported[key].items()]
    for r in rows:
        r["equal"] = r["measured"] == r["reported"]
    return rows


def run_mode(root: Path, out: Path, mode: str, seeds: Sequence[int], device: str,
             categories: Sequence[str] | None) -> None:  # pragma: no cover - GPU
    from vlmab.datasets.mvtec_ad import MVTecAD
    from vlmab.eval.provenance import method_meta
    from vlmab.eval.runner import run_evaluation
    from vlmab.eval.store import ResultStore
    from vlmab.methods.patchcore_backend import PatchCoreBackend
    from vlmab.methods.patchcore_ref import PatchCoreRef

    dataset = MVTecAD(root)
    store = ResultStore(out / "shards")
    for seed in seeds:
        for category in categories or dataset.categories():
            # A fresh backend per CATEGORY, as run_mvtec_ad.py: one backend reused across
            # categories fits only the first (anomalib 2.6.0; MEASURED 2026-10-21).
            method = PatchCoreRef(backend=PatchCoreBackend(
                seed=seed, preprocess=PREPROCESS, diagnostic_aliased_forward=MODES[mode]))
            meta = method_meta(method, {"method": "patchcore_ref", "split": "test",
                                        "experiment": "gate_antialias_ablation"})
            run_evaluation(dataset, method, store, meta, categories=[category], split="test",
                           fit_split="train", maps_dir=out / "maps", device=device)
        print(f"  {mode} seed {seed}: done (resize_antialias {not MODES[mode]})")


def score_mode(results: Path, mode: str) -> dict:  # pragma: no cover - needs the shards
    from reproduction_gate import main as gate
    from vlmab.eval.store import ResultStore

    shards = results / mode / "shards"
    check_mode(ResultStore(shards).load_all(), mode)
    out = results / mode / "gate.md"
    code = gate(["--results", str(shards), "--targets", str(TARGETS), "--all-seeds",
                 "--out", str(out)])
    if code == 2 or not out.is_file():
        raise SystemExit(f"{mode}: the gate refused to score {shards}; read the output above")
    return parse_gate_report(out.read_text())


def print_comparison(mode: str, measured: dict) -> None:
    rows = compare_reports(measured, parse_gate_report(REPORTS[mode].read_text()))
    print(f"\n== {mode} (this VM) vs the committed {REPORTS[mode]}")
    print(f"  {'':<14}{'this VM':>14}{'report':>14}  same")
    for r in rows:
        print(f"  {r['what']:<14}{r['measured']:>14}{r['reported']:>14}  "
              f"{'yes' if r['equal'] else 'NO'}")
    print(f"  {sum(r['equal'] for r in rows)} of {len(rows)} equal at the report's decimals")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=sorted(MODES), default=None,
                        help="run this mode only (default: both, aliased first)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--categories", nargs="+", default=None,
                        help="a subset is a probe; the comparison needs all fifteen")
    parser.add_argument("--root", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compare-only", action="store_true",
                        help="run nothing: score both modes' shards and compare")
    args = parser.parse_args(argv)

    head = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse",
                           "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"{Path(__file__).name} running from commit {head or 'unknown'}")
    try:
        import torch
        print(f"torch {torch.__version__}, CUDA available {torch.cuda.is_available()}\n")
    except ImportError:
        print("torch not importable\n")

    if not args.compare_only:
        for mode in [args.mode] if args.mode else list(MODES):
            print(f"[{mode}] MVTec AD classic, seeds {args.seeds}")
            run_mode(args.root, args.results / mode, mode, args.seeds, args.device,
                     args.categories)
        if args.mode or args.categories:
            return 0       # one (mode, seed) or a probe: comparing is a separate call

    measured = {mode: score_mode(args.results, mode) for mode in MODES}
    for mode in MODES:
        print_comparison(mode, measured[mode])

    a, b = measured["aliased"], measured["antialiased"]
    print("\n== The correction on this VM: aliased -> antialiased, I-AUROC (%)")
    for name in a["categories"]:
        x, y = (float(m["categories"][name].split(" ± ")[0]) for m in (a, b))
        print(f"  {name:<14}{x:>8.2f} -> {y:>6.2f}  ({y - x:+.2f})")
    print(f"  {'gate mean':<14}{float(a['mean']):>8.2f} -> {float(b['mean']):>6.2f}  "
          f"({float(b['mean']) - float(a['mean']):+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
