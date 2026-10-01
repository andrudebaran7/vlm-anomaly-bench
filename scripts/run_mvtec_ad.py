#!/usr/bin/env python3
"""Run one method over MVTec AD classic's 15 categories: the reproduction gate's data path.

    python scripts/run_mvtec_ad.py --method winclip
    python scripts/run_mvtec_ad.py --method winclip --categories bottle cable   # a probe

**This existed only as notebook cells until 2026-10-01.** PatchCore's gate ran from cells 3.2 and
3b.1, which is why phase 3b had to be re-typed by hand and why none of its fit times were
recorded. WinCLIP needs this path for its MVTec AD gate (91.8 ± 1.0, protocol §2 v0.2.14) and
`run_visa_secondary.py` for its VisA one; both of WinCLIP's figures gate, unlike PatchCore's.

**All fifteen categories are the default** because the published figure every gate compares
against is their mean. A subset is a probe and says so rather than producing a number that looks
like a verdict.

**Scoring is `scripts/reproduction_gate.py`, which this script does not call.** The gate reads
shards, so running and scoring stay separable: a run that crashes at category twelve can still be
scored over the eleven it finished, and a scoring bug costs no GPU time to fix.

**No Drive machinery.** MVTec AD classic downloads directly, so there is no archive behind a
registration form, nothing to fetch from Drive and nothing to restore.

Exit codes: **0 ran, 1 a run failed, 2 could not start** (unknown method, repeated seed).
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

#: Protocol §6: three seeds wherever stochasticity exists. Nothing here decides that a method is
#: deterministic -- that is a measurement, and WinCLIP's targets file pre-registers how to make
#: it. Fewer than three warns and continues, the same as the AD 2 runner.
DEFAULT_SEEDS = (0, 1, 2)

#: The published means every gate compares against are over all fifteen.
N_CATEGORIES = 15


def default_results(method: str) -> Path:
    """Method-scoped, for the reason `run_mvtec_ad2.default_results` documents: the gate reads
    every shard in the directory, and two methods there would be scored as one."""
    return Path("results/reproduction") / method / "mvtec_ad"


def run_one_seed(root: Path, results: Path, seed: int, device: str, method_name: str,
                 categories: Sequence[str]) -> None:
    """Score every category at one seed. Needs anomalib and a GPU; imported here for that."""
    from vlmab.datasets.mvtec_ad import MVTecAD
    from vlmab.eval.provenance import run_meta
    from vlmab.eval.runner import run_evaluation
    from vlmab.eval.store import ResultStore
    from vlmab.methods.gpu import build_runnable

    # A fresh adapter per seed, for the reason the AD 2 runner documents: reusing one carries the
    # previous seed's state into the next seed's scores while the shard claims otherwise.
    method = build_runnable(method_name, seed=seed)
    store = ResultStore(results / "shards")
    # `preprocess` decides a run's identity for PatchCore and does not exist for a zero-shot
    # method. None records that; a fabricated value would claim a property the method lacks.
    preprocess = getattr(method, "preprocess", None)
    meta = run_meta({"method": method_name, "split": "test", "preprocess": preprocess})
    meta = {**meta, "preprocess": preprocess}

    run_evaluation(
        MVTecAD(root), method, store, meta,
        categories=list(categories), split="test", fit_split="train",
        maps_dir=results / "maps", device=device,
    )
    print(f"  seed {seed}: done over {len(categories)} categories (preprocess {preprocess})")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", default="patchcore_ref",
                        help="a method with a GPU backend; see vlmab.methods.gpu")
    parser.add_argument("--root", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument("--results", type=Path, default=None,
                        help="default: results/reproduction/<method>/mvtec_ad")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="a subset makes this a probe, not a gate run")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    # Printed FIRST, before anything can fail: a stale clone is invisible in a traceback, and one
    # live run was lost to exactly that (2026-09-21).
    here = Path(__file__).resolve().parent.parent
    head = subprocess.run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip() or "unknown"
    print(f"{Path(__file__).name} running from commit {head}\n")

    # Checked before anything is read from disk: a wrong method must cost nothing.
    from vlmab.methods.gpu import gpu_builders
    if args.method not in gpu_builders():
        print(f"\ncannot start: no GPU backend for {args.method!r}; "
              f"methods with one: {gpu_builders()}", file=sys.stderr)
        return 2

    if len(args.seeds) != len(set(args.seeds)):
        print(f"repeated seed in {args.seeds}; each seed is one run", file=sys.stderr)
        return 2

    from vlmab.datasets.mvtec_ad import MVTecAD
    categories = args.categories or MVTecAD(args.root).categories()
    if len(categories) != N_CATEGORIES:
        print(f"⚠️  {len(categories)} categories, not {N_CATEGORIES}. The published figure every "
              "gate compares against is the mean over all of them, so this is a probe and its "
              "numbers are not a gate verdict.\n")
    if len(args.seeds) < 3:
        print(f"⚠️  {len(args.seeds)} seed(s). Protocol §6 requires three for any REPORTED "
              "number wherever stochasticity exists; anything less is a measurement.\n")

    results = args.results or default_results(args.method)
    for i, seed in enumerate(args.seeds, 1):
        print(f"[seed {i}/{len(args.seeds)}] {args.method} over MVTec AD classic at seed {seed}")
        run_one_seed(args.root, results, seed, args.device, args.method, categories)

    print(f"\n{args.method} done at seeds {args.seeds}. Shards: {results / 'shards'}\n"
          "Score it with reproduction_gate.py; the target block for mvtec_ad is the one "
          "vlmab.eval.targets.block_for finds in "
          f"configs/reproduction/{args.method}.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
