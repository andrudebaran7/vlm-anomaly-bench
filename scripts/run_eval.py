#!/usr/bin/env python
"""Run one method over a dataset, writing per-category result shards.

    python scripts/run_eval.py --method intensity_baseline --root data/mvtec_ad2 \
        --split test_public --results results/shards --seed 0
"""
import argparse
import sys
from pathlib import Path
from typing import Sequence

from vlmab.datasets.mvtec_ad2 import MVTecAD2
from vlmab.eval.provenance import run_meta
from vlmab.eval.runner import run_evaluation
from vlmab.eval.store import ResultStore
from vlmab.methods.registry import available, build_method


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--split", default="test_public")
    parser.add_argument("--category", default=None)
    parser.add_argument("--maps-dir", default=None, type=Path)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    try:
        method = build_method(args.method)
    except KeyError as exc:
        print(exc.args[0])
        print(f"available methods: {available()}")
        return 1

    dataset = MVTecAD2(args.root)
    # Resolved before ResultStore exists: ResultStore.__init__ creates --results on disk, and
    # if --results is nested inside --root (as it legitimately can be, e.g. in a tmp-dir test),
    # dataset.categories() called any later would pick up that now-existing, category-less
    # output directory as a bogus category and fail scanning it.
    categories = [args.category] if args.category else dataset.categories()
    store = ResultStore(args.results)
    meta = run_meta({"method": args.method, "split": args.split}, seed=args.seed)

    # method.prepare() is NOT called here: run_evaluation() owns that decision, calling it
    # only if there is genuinely work left (see its docstring) so a fully-resumed run never
    # pays for a model load and a run with work never loads it twice.
    try:
        written = run_evaluation(
            dataset, method, store, meta,
            categories=categories,
            split=args.split,
            maps_dir=args.maps_dir,
            device=args.device,
        )
    except RuntimeError as exc:
        # The specific "this adapter cannot run in this environment" signal (e.g. an MLLM
        # adapter with no injected/constructible model client) -- fail cleanly instead of
        # an uncaught traceback. Any other exception type is a real bug and should propagate.
        print(f"{args.method}: cannot run here ({exc})")
        return 1

    print(f"{args.method}: wrote {len(written)} shard(s) to {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
