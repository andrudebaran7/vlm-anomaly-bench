#!/usr/bin/env python3
"""Export the image-level scores behind the intervals, from the Drive shards, for the repository.

The shards also hold local paths and run metadata and stay on Drive. This writes only what
`scripts/bootstrap_iauroc.py` and `scripts/scene_sensitivity.py` read, one CSV per configuration and
category:

    <out>/<config>/<category>.csv    folder,index,condition,label,seed,score

Each configuration must first reproduce its committed report, the same refusal the bootstrap applies.
Fruit Jelly's aliased shards are gone, so it has no aliased file. Scores keep full float precision, so
the intervals recomputed from these files are the ones computed from the shards.

    python scripts/export_image_scores.py \\
        --antialiased /content/drive/MyDrive/mvtec_ad2_results/patchcore_ref_v0.2.21 \\
        --aliased /content/drive/MyDrive/mvtec_ad2_results \\
        --zip /content/drive/MyDrive/mvtec_ad2_results/image_scores_export.zip
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_iauroc import CATEGORIES, REPORTS, Paired, check_against_report, load  # noqa: E402

COLUMNS = ["folder", "index", "condition", "label", "seed", "score"]


def write_scores(table: pd.DataFrame, path: Path) -> None:
    """One image_table as a CSV, in a stable order, with no column beyond COLUMNS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table[COLUMNS].sort_values(COLUMNS[:5]).to_csv(path, index=False, lineterminator="\n")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs the shards
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--antialiased", type=Path, required=True)
    parser.add_argument("--aliased", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/image_scores"))
    parser.add_argument("--zip", type=Path, default=None)
    args = parser.parse_args(argv)

    written = 0
    for config, root in (("antialiased", args.antialiased), ("aliased", args.aliased)):
        for category in CATEGORIES:
            try:
                table = load(root, category, config)
            except FileNotFoundError as exc:
                if config == "antialiased":
                    raise
                print(f"{config}/{category}: {exc} -- no file")
                continue
            bad = check_against_report(Paired([table]), config,
                                       (REPORTS[config] / f"{category}.md").read_text())
            if bad:
                print(f"REFUSED {config}/{category}: shards do not reproduce the report: {bad}")
                return 2
            path = args.out / config / f"{category}.csv"
            write_scores(table, path)
            written += 1
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            print(f"{config}/{category}: {len(table)} rows, {path.stat().st_size} bytes, sha256 {digest}")
    print(f"\n{written} files under {args.out}")
    if args.zip is not None:
        made = shutil.make_archive(str(args.zip.with_suffix("")), "zip", args.out)
        print(f"wrote {made} ({Path(made).stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
