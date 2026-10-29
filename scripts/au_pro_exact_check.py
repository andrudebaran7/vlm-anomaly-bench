#!/usr/bin/env python3
"""AU-PRO on a real run's maps, from the reported 200-point grid and from the exact sweep.

Review item #24 (2026-10-29): the grid's error was measured on synthetic float16 maps (-3.5e-4
at 0.3) but never on an MVTec AD 2 category, because the maps never leave the Colab VM. This
runs there, on a results root the runner just wrote (`shards/` plus the maps they point at), and
prints, per lighting condition and FPR limit, both values and their difference -- and whether
the difference changes the value at the three decimals the paper prints.

    python scripts/au_pro_exact_check.py --results results/au_pro_check/sheet_metal

One seed is enough: this measures the metric, not the anchor. Memory: one lighting condition's
maps and masks at a time, as `aggregate` holds them.
"""
import argparse
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Sequence

from vlmab.eval.aggregate import _load_pair
from vlmab.eval.store import ResultStore
from vlmab.metrics.pixel_level import au_pro

LIMITS = (0.3, 0.05)


def _three(x: float) -> str:
    return str(Decimal(repr(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def check(results: Path) -> list[dict]:
    df = ResultStore(Path(results) / "shards").load_all()
    if df.empty:
        raise FileNotFoundError(f"no shards under {Path(results) / 'shards'}")
    seeds = sorted({str(s) for s in df["seed"].unique()})
    if len(seeds) != 1:
        raise ValueError(f"one seed per check, found {seeds}; point --results at a one-seed run")
    df = df[df["map_path"].notna()]
    rows = []
    for lighting, group in sorted(df.groupby("meta_lighting"), key=lambda kv: kv[0]):
        masks, amaps = zip(*(_load_pair(r) for r in group.itertuples()))
        for limit in LIMITS:
            grid = au_pro(masks, amaps, limit)
            exact = au_pro(masks, amaps, limit, exact=True)
            rows.append({"lighting": lighting, "limit": limit, "n": len(masks),
                         "grid": grid, "exact": exact, "diff": grid - exact})
        del masks, amaps
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, type=Path,
                        help="a results root with shards/ whose map_path files exist here")
    args = parser.parse_args(argv)

    rows = check(args.results)
    print(f"{'lighting':<14}{'limit':>6}{'n':>4}{'grid':>12}{'exact':>12}{'grid-exact':>13}  3 dp")
    for r in rows:
        same = _three(r["grid"]) == _three(r["exact"])
        print(f"{r['lighting']:<14}{r['limit']:>6}{r['n']:>4}{r['grid']:>12.6f}{r['exact']:>12.6f}"
              f"{r['diff']:>+13.2e}  {'same' if same else _three(r['grid']) + ' vs ' + _three(r['exact'])}")
    worst = max(rows, key=lambda r: abs(r["diff"]))
    changed = sum(_three(r["grid"]) != _three(r["exact"]) for r in rows)
    print(f"\nmax |grid - exact| = {abs(worst['diff']):.2e} ({worst['lighting']}, limit {worst['limit']})")
    print(f"values whose 3-decimal rounding differs: {changed} of {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
