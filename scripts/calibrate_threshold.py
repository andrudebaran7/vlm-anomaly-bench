#!/usr/bin/env python3
"""Fit the pre-registered threshold rules on a defect-free `validation` run.

The output is committed: it *is* the pre-registration required by protocol §4 (v0.2.11).
Run it once per method, after that method's validation run, and commit the YAML before any
private-split submission.

    run_eval.py --split validation --results runs/val --maps-dir runs/val/maps --method M ...
    calibrate_threshold.py --results runs/val --dataset mvtec_ad2 --method M \\
        --out configs/thresholds/mvtec_ad2__M.yaml
    git add configs/thresholds/mvtec_ad2__M.yaml && git commit

The alpha-sensitivity curve protocol §4 requires is produced by running this at each point of
the registered grid {1e-4, 1e-3, 1e-2, 1e-1} into *throwaway* paths, then scoring each one with
`aggregate.threshold_metrics`. Only the alpha=1e-3 artifact is committed: the sweep is a
reported curve, not a choice, and an uncommitted artifact cannot be mistaken for one.
"""
import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from vlmab.eval.store import ResultStore
from vlmab.threshold.artifact import write_artifact
from vlmab.threshold.rules import Calibration, calibrate

#: Protocol §4 (v0.2.11). Both are pre-registered, so they are defaults rather than choices
#: a caller is expected to make -- overriding them is what the sensitivity sweep does, and it
#: does not overwrite the committed artifact.
PREREGISTERED_ALPHA = 1e-3
DESIGNATED_RULE = "per_image_robust_z"

#: Fitting a threshold on anything else is calibrating on test data.
CALIBRATION_SPLIT = "validation"


def _run_id_from_maps(map_paths) -> str:
    """Recover the run from the map directory name.

    The runner names it `<dataset>__<method>__<run_id>__<category>__<seed_tag>` (the seed_tag
    suffix was added so two seeds of one config never share a directory) and does **not** write
    a `run_id` shard column, so this directory name is the only record of which run produced
    these maps -- and the artifact is worth much less without it.
    """
    ids = set()
    for name in {Path(p).parent.name for p in map_paths}:
        parts = name.split("__")
        if len(parts) != 5:
            raise ValueError(
                f"map directory {name!r} does not match the runner's "
                "<dataset>__<method>__<run_id>__<category>__<seed_tag> layout, so the run this "
                "calibration came from cannot be recorded"
            )
        ids.add(parts[2])
    return ",".join(sorted(ids))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="ResultStore root of the validation run")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--alpha", default=PREREGISTERED_ALPHA, type=float)
    parser.add_argument("--designate", default=DESIGNATED_RULE)
    args = parser.parse_args(argv)

    if args.designate == "transductive_quantile":
        raise ValueError(
            "--designate transductive_quantile is not allowed: protocol §4 (v0.2.11) fixes "
            "per_image_robust_z as the designated submission rule specifically because "
            "transductive_quantile adapts to the split it is scoring, which a pre-registered "
            "threshold must not do"
        )

    df = ResultStore(args.results).load_all()
    if df.empty:
        raise ValueError(f"no shards under {args.results}")
    df = df[(df["dataset"] == args.dataset) & (df["method"] == args.method)]
    if df.empty:
        raise ValueError(
            f"no shard for dataset={args.dataset!r} method={args.method!r} under {args.results}"
        )

    # Shards are keyed (dataset, method, category) with no split, so a test_public run written
    # to this root is indistinguishable by filename. Check the column, not the caller's word.
    #
    # A shard written without a `split` column at all is a real path: ResultStore.load_all()
    # concatenates every shard under the root, and pandas fills a column a given shard lacks
    # with NaN for that shard's rows. Sorting a set that mixes a str and a float NaN raises
    # TypeError, not a useful ValueError -- so NaN is named explicitly rather than sorted raw.
    if "split" not in df.columns:
        raise ValueError(
            f"the shard(s) for dataset={args.dataset!r} method={args.method!r} under "
            f"{args.results} carry no split column, so this run cannot be shown to be a "
            f"{CALIBRATION_SPLIT!r} run; refusing to calibrate (protocol §4)"
        )
    found = sorted(
        ({s if isinstance(s, str) else "NaN" for s in df["split"].unique()}), key=str
    )
    if found != [CALIBRATION_SPLIT]:
        raise ValueError(
            f"refusing to calibrate on split(s) {found}: a threshold rule must be fitted on "
            f"{CALIBRATION_SPLIT!r}, which is defect-free, never on test data (protocol §4). "
            f"Point --results at a run made with --split {CALIBRATION_SPLIT}."
        )

    # A threshold is fitted on the maps one run produced. Two seeds pooled fits it on a mixture
    # no single run ever produces, and calibration would still succeed and write an artifact
    # that looks pre-registered. Same refusal the metric functions make.
    if "seed" in df.columns:
        seeds = sorted({None if pd.isna(s) else s for s in df["seed"].unique()}, key=str)
        if len(seeds) > 1:
            raise ValueError(
                f"the shards under {args.results} pool {len(seeds)} seeds ({seeds}): a threshold "
                "is calibrated on one run's maps, so pooling seeds fits it on a distribution no "
                "single run produced. Calibrate each seed separately (protocol §4)."
            )
    if "map_path" not in df.columns or df["map_path"].isna().all():
        raise ValueError(
            f"the run under {args.results} saved no anomaly maps (no usable map_path column); "
            "re-run it with --maps-dir, since calibration reads the maps themselves"
        )

    all_maps = [p for p in df["map_path"] if isinstance(p, str)]
    categories: dict[str, dict[str, Calibration]] = {}
    for category, group in sorted(df.groupby("category"), key=lambda kv: kv[0]):
        paths = [p for p in group["map_path"] if isinstance(p, str)]
        factory = lambda paths=paths: (np.load(p).astype(np.float32) for p in paths)
        categories[category] = {
            "global_quantile": calibrate("global_quantile", factory, args.alpha),
            "per_image_robust_z": calibrate("per_image_robust_z", factory, args.alpha),
            "transductive_quantile": Calibration(
                "transductive_quantile", float("nan"), args.alpha, 0, 0
            ),
        }

    path = write_artifact(
        args.out,
        dataset=args.dataset,
        method=args.method,
        alpha=args.alpha,
        calibrated_on={
            "split": CALIBRATION_SPLIT,
            "n_images": int(len(df)),
            "lighting": sorted(set(df["meta_lighting"])) if "meta_lighting" in df.columns else [],
        },
        run_id=_run_id_from_maps(all_maps),
        categories=categories,
        designated_for_submission=args.designate,
    )
    if args.alpha != PREREGISTERED_ALPHA:
        print(
            f"warning: --alpha {args.alpha} differs from the pre-registered "
            f"{PREREGISTERED_ALPHA}; this is a sensitivity-sweep artifact and must not be "
            "committed as the pre-registration (protocol §4)",
            file=sys.stderr,
        )
        print(f"wrote {path}")
    else:
        print(f"wrote {path} — commit it before any submission (protocol §4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
