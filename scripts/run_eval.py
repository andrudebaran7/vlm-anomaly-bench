#!/usr/bin/env python
"""Run one method over a dataset, writing per-category result shards.

    python scripts/run_eval.py --method intensity_baseline --root data/mvtec_ad2 \
        --split test_public --results results/shards

    # a reproduction run against MVTec AD classic (protocol §2), whose split is "test"
    python scripts/run_eval.py --method patchcore_ref --dataset mvtec_ad \
        --root data/mvtec_ad --split test --results runs/repro --seed 0
"""
import argparse
import sys
from pathlib import Path
from typing import Sequence

from vlmab.datasets.registry import available as available_datasets, build_dataset
from vlmab.eval.provenance import run_meta
from vlmab.eval.runner import run_evaluation
from vlmab.eval.store import ResultStore
from vlmab.methods.base import MethodNotRunnable
from vlmab.methods.registry import available, build_method


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--dataset", default="mvtec_ad2",
        help="which loader reads --root; the name is stamped into every shard and is what\n"
             "the reproduction gate matches its targets against",
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--split", default="test_public")
    parser.add_argument("--category", default=None)
    parser.add_argument("--maps-dir", default=None, type=Path)
    parser.add_argument(
        "--seed", default=None, type=int,
        help="seed to apply to a stochastic method; omit for a method that seeds nothing "
             "(recorded as unseeded, which is the truth)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    try:
        method = build_method(args.method, seed=args.seed)
    except KeyError as exc:
        print(exc.args[0])
        print(f"available methods: {available()}")
        return 1
    except ValueError as exc:
        # A method that cannot apply the seed it was handed. Report it the way an unknown
        # method is reported, rather than a traceback: the run is refused, not broken.
        print(f"{args.method}: {exc}")
        return 1

    try:
        dataset = build_dataset(args.dataset, args.root)
    except KeyError as exc:
        print(exc.args[0])
        print(f"available datasets: {available_datasets()}")
        return 1
    # Resolved before ResultStore exists: ResultStore.__init__ creates --results on disk, and
    # if --results is nested inside --root (as it legitimately can be, e.g. in a tmp-dir test),
    # dataset.categories() called any later would pick up that now-existing, category-less
    # output directory as a bogus category and fail scanning it.
    categories = [args.category] if args.category else dataset.categories()
    store = ResultStore(args.results)
    meta = run_meta({"method": args.method, "split": args.split})

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
    except MethodNotRunnable as exc:
        # The specific "this adapter cannot run in this environment" signal (e.g. an MLLM
        # adapter with no injected/constructible model client) -- fail cleanly instead of
        # an uncaught traceback. Any other exception type (including a plain RuntimeError
        # raised from inside a real predict()) is a real bug and must propagate.
        print(f"{args.method}: cannot run here ({exc})")
        return 1

    print(f"{args.method}: wrote {len(written)} shard(s) to {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
