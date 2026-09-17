#!/usr/bin/env python3
"""Run PatchCore over VisA's 12 objects and score the secondary check (protocol §2 v0.2.12).

    python scripts/run_visa_secondary.py            # fetch if needed, run, score
    python scripts/run_visa_secondary.py --seed 1   # a second seed into the same root

VisA is a **secondary check**: it is reported with its caveat and cannot fail the method.
PatchCore's own paper predates VisA and reports no VisA number, so the only primary source is the
VisA dataset paper's Table 6 (92.4, 1-class, 12 objects), which states none of its PatchCore
hyperparameters — a miss here cannot distinguish a wrong implementation from a different setup.

**Why this is a script and not a notebook cell.** A fix inside notebook JSON cannot reach a live
Colab session: cell 1.1 hard-resets the repository clone, while the notebook the session is
executing is a different file, so a changed cell costs a full reload (reinstalling anomalib,
re-downloading the backbone). A tracked `.py` is fetched by a cell the session already has. Cell
3.4 therefore calls this, and this is where the logic lives.

**Cost.** VisA's one-class split carries 8,659 train normals across 12 objects, against MVTec AD
classic's 3,629 across 15, plus 2,162 test images. Budget roughly two hours on a T4. One object is
the resume unit: re-running skips whatever already has a shard.

**§6 still applies.** "Reported, never a gate" means a miss cannot fail the method; it does not
exempt the number from the three-seed rule. One seed is a first reading, not a table entry.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

#: Byte-exact size of `VisA_20220922.tar`, measured by HTTP HEAD 2026-07-23 and confirmed on
#: download 2026-09-16 (docs/datasets-access.md). A truncated download is the likeliest way to
#: start a two-hour run on data that is silently incomplete, so the size is checked rather than
#: trusted -- and `tests/test_visa_secondary_script.py` holds this constant to that record.
ARCHIVE_BYTES = 1_929_840_640
ARCHIVE_URL = (
    "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
)
#: `classic` was refuted on 2026-09-17 (docs/next-steps.md); every number in this repo comes from
#: the anomalib transform and this check has no reason to differ from the runs it is read beside.
PREPROCESS = "anomalib"


def fetch(root: Path) -> None:
    """Download and extract VisA unless its split file is already there.

    CC BY 4.0 on AWS Open Data: no account, no form, no Drive round trip. `split_csv/1cls.csv` is
    the marker because it is what the loader refuses to run without -- the 90/10 normal split
    cannot be reconstructed from the directory tree.
    """
    if (root / "split_csv" / "1cls.csv").is_file():
        print(f"VisA already extracted under {root}")
        return

    root.mkdir(parents=True, exist_ok=True)
    archive = root / "VisA_20220922.tar"
    print(f"fetching {ARCHIVE_URL} ({ARCHIVE_BYTES / 2**30:.2f} GiB)...")
    subprocess.run(["curl", "-sL", "-o", str(archive), ARCHIVE_URL], check=True)

    size = archive.stat().st_size
    if size != ARCHIVE_BYTES:
        raise ValueError(
            f"downloaded {size:,} bytes, expected {ARCHIVE_BYTES:,}. A truncated archive would "
            "produce a run over incomplete data that looks entirely normal. Delete "
            f"{archive} and retry."
        )
    subprocess.run(["tar", "-xf", str(archive), "-C", str(root)], check=True)
    archive.unlink()
    print(f"extracted to {root}")


def run(root: Path, results: Path, seed: int | None, device: str) -> None:
    """Fit and score one object at a time. Needs anomalib and a GPU; imported here for that."""
    from vlmab.datasets.visa import VisA
    from vlmab.eval.provenance import run_meta
    from vlmab.eval.runner import run_evaluation
    from vlmab.eval.store import ResultStore
    from vlmab.methods.patchcore_backend import PatchCoreBackend
    from vlmab.methods.patchcore_ref import PatchCoreRef

    visa = VisA(root)
    store = ResultStore(results / "shards")
    meta = run_meta({"method": "patchcore_ref", "split": "test", "preprocess": PREPROCESS})

    categories = visa.categories()
    print(f"{len(categories)} objects: {categories}")
    for i, cat in enumerate(categories, 1):
        # A fresh backend per object: the memory bank is per-category by construction, and
        # reusing one would carry the previous object's bank over.
        method = PatchCoreRef(backend=PatchCoreBackend(seed=seed, preprocess=PREPROCESS))
        # `preprocess` is stamped from the method, not from the constant above: the value
        # recorded has to be the one that was applied.
        run_evaluation(
            visa, method, store, {**meta, "preprocess": method.preprocess},
            categories=[cat], split="test", fit_split="train",
            maps_dir=results / "maps", device=device,
        )
        print(f"[{i}/{len(categories)}] {cat}: done "
              f"(seed {method.seed}, preprocess {method.preprocess})")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/visa"))
    parser.add_argument("--results", type=Path, default=Path("results/reproduction/visa"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path,
                        default=Path("results/reproduction/patchcore_visa.md"))
    parser.add_argument("--skip-fetch", action="store_true",
                        help="the data is already in place; go straight to running")
    parser.add_argument("--no-score", action="store_true",
                        help="run the objects but do not score (e.g. more seeds are coming)")
    args = parser.parse_args(argv)

    if not args.skip_fetch:
        fetch(args.root)
    run(args.root, args.results, args.seed, args.device)

    if args.no_score:
        print("skipping the score step as asked")
        return 0

    # Scored through the gate rather than inline, so the secondary check goes through exactly the
    # same refusals as the gate does -- wrong dataset, wrong method, incomplete object set.
    gate = Path(__file__).resolve().parent / "reproduction_gate.py"
    return subprocess.run(
        [sys.executable, str(gate),
         "--results", str(args.results / "shards"),
         "--targets", "configs/reproduction/patchcore_ref.yaml",
         "--which", "secondary",
         "--out", str(args.out)],
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
