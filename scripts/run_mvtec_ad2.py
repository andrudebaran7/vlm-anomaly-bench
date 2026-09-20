#!/usr/bin/env python3
"""Run one MVTec AD 2 category through PatchCore, at every seed protocol §6 requires (M3).

    # In a notebook cell FIRST — this script cannot mount Drive, see fetch():
    #     from google.colab import drive; drive.mount('/content/drive')
    python scripts/run_mvtec_ad2.py --category vial
    python scripts/run_mvtec_ad2.py --category vial --seeds 0   # a measurement, not a result

**One category is the whole unit of work here**, and that is not a style choice: it is
simultaneously the download unit, the Drive-upload unit, the resume unit and the shard unit.
MVTec AD 2 is 30.4 GB against Google Drive's free 15 GB, and the archives sit behind mvtec.com's
registration form, so a session cannot fetch them unattended. Each category is uploaded to
`MyDrive/mvtec_ad2/<category>.tar.gz` once, by hand, and discarded from Drive when its shards
exist. Peak disk is then the largest single category (Fabric, 10 GB), not the dataset.

**Every seed runs before the next category starts**, rather than the whole grid at seed 0 and
then again at seed 1. The alternative would need each category on Drive three separate times,
which is the one resource this study actually cannot buy more of.

**This does not score the `validation` split**, though the data is right there and a later M4
calibration will need it. Two reasons, both deliberate: `run_evaluation` re-fits per call, so a
second split would roughly double the fit cost per seed; and the threshold rule's seed semantics
— one calibration per seed, or one for the seed that gets submitted — are not pre-registered yet,
so producing those scores now risks producing the wrong ones. The cost of that decision is a
second upload of each category when M4 runs, and it is recorded rather than absorbed silently.

**Shards go to Drive as each seed finishes**, not at the end. A four-hour run that loses its VM
at the last category loses everything otherwise; that lesson is from 2026-09-19 and cost a whole
session's worth of nerves (`scripts/save_visa_to_drive.py` exists for the same reason).

Exit codes: **0 ran, 1 a run failed, 2 could not start** (archive missing, layout wrong) — so a
category that was never on Drive is never mistaken for one that produced no anomalies.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

#: Protocol §6: three seeds wherever stochasticity exists, and PatchCore's coreset sampling is
#: stochastic — measured 2026-08-26, and the reason the whole seed-provenance apparatus exists.
DEFAULT_SEEDS = (0, 1, 2)
#: `classic` was refuted 2026-09-17. Every number in this repo comes from the anomalib transform.
PREPROCESS = "anomalib"
DRIVE_ARCHIVES = Path("/content/drive/MyDrive/mvtec_ad2")
DRIVE_RESULTS = Path("/content/drive/MyDrive/mvtec_ad2_results")


class DriveNotMounted(RuntimeError):
    """Raised instead of attempting a mount this process cannot perform."""


def fetch(category: str, root: Path, archives: Path) -> None:
    """Extract <category>.tar.gz from an ALREADY-MOUNTED Drive, unless it is already on disk.

    **This deliberately does not mount Drive, because it cannot.** `google.colab.drive.mount`
    sends an authentication request to the Colab frontend *through the IPython kernel*; in a
    subprocess started by `!python` there is no kernel, `get_ipython()` returns None, and the
    call dies inside Colab's own `_message.send_request` with an AttributeError that names
    nothing relevant. Mounting is the notebook's job, in the kernel; reading a mounted Drive is
    a plain filesystem operation and works fine from here.
    """
    if (root / category / "train").is_dir():
        print(f"{category} already extracted at {root / category} — skipping")
        return

    tar = archives / f"{category}.tar.gz"
    if not tar.is_file():
        # Two very different problems that would otherwise produce the same message.
        drive_root = Path("/content/drive")
        if str(archives).startswith(str(drive_root)) and not (drive_root / "MyDrive").is_dir():
            raise DriveNotMounted(
                "Google Drive is not mounted, so nothing under /content/drive exists yet.\n"
                "This script cannot mount it: drive.mount() talks to the Colab frontend through\n"
                "the IPython kernel, and a `!python` subprocess has no kernel.\n\n"
                "Run this in a notebook cell, then re-run this script:\n\n"
                "    from google.colab import drive; drive.mount('/content/drive')\n"
            )
        raise FileNotFoundError(
            f"{tar} not found. Upload {category}.tar.gz to {archives}/ and re-run. The archive is "
            "the one from the MVTec download page, unmodified — it sits behind a registration "
            "form, so this cannot be fetched automatically."
        )

    root.mkdir(parents=True, exist_ok=True)
    print(f"extracting {tar} ({tar.stat().st_size / 2**30:.2f} GiB)...")
    # The archive's top level is <category>/, so -C <root> needs no path surgery (verified
    # against the real vial.tar.gz, docs/datasets-access.md).
    subprocess.run(["tar", "-xzf", str(tar), "-C", str(root)], check=True)
    print(f"extracted to {root / category}")


def verify_layout(category: str, root: Path) -> None:
    """The layout check that already exists, rather than a second opinion invented here."""
    script = Path(__file__).resolve().parent / "prepare_data.py"
    result = subprocess.run([sys.executable, str(script), "--root", str(root),
                             "--category", category])
    if result.returncode != 0:
        raise RuntimeError(
            f"{script.name} rejected {root / category}. A run over a malformed category would "
            "look entirely normal in its output."
        )


def run_one_seed(category: str, root: Path, results: Path, seed: int, device: str) -> None:
    """Fit and score one category at one seed. Needs anomalib and a GPU; imported here for that."""
    from vlmab.datasets.mvtec_ad2 import MVTecAD2
    from vlmab.eval.provenance import run_meta
    from vlmab.eval.runner import run_evaluation
    from vlmab.eval.store import ResultStore
    from vlmab.methods.patchcore_backend import PatchCoreBackend
    from vlmab.methods.patchcore_ref import PatchCoreRef

    # A fresh backend per seed: the memory bank is what the seed changes, and reusing one would
    # carry the previous seed's bank into the next seed's scores while the shard claimed
    # otherwise. This is the defect the seed-provenance work already had to undo once.
    method = PatchCoreRef(backend=PatchCoreBackend(seed=seed, preprocess=PREPROCESS))
    store = ResultStore(results / "shards")
    # run_meta HASHES its cfg into config_hash; it does not pass the keys through as columns.
    # So `preprocess` is added on top, as a real column, exactly as the VisA runner does --
    # otherwise the value that decides a run's identity would be recoverable only by recomputing
    # a hash. Stamped from the method, not from the constant above: what is recorded has to be
    # what was applied.
    meta = run_meta({"method": "patchcore_ref", "split": "test_public",
                     "preprocess": method.preprocess})
    meta = {**meta, "preprocess": method.preprocess}

    run_evaluation(
        MVTecAD2(root), method, store, meta,
        categories=[category], split="test_public", fit_split="train",
        maps_dir=results / "maps", device=device,
    )
    print(f"  {category} seed {seed}: done (preprocess {method.preprocess})")


def save_to_drive(results: Path, dest: Path) -> None:
    """Copy the shards off the VM. The maps stay: they are large and regenerable."""
    shards = results / "shards"
    if not shards.is_dir() or not any(shards.glob("*.parquet")):
        print("nothing to copy yet")
        return
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(shards, dest / "shards", dirs_exist_ok=True)
    n = len(list((dest / "shards").glob("*.parquet")))
    print(f"  saved to {dest / 'shards'} ({n} shard(s) there now)")


def summarise(results: Path) -> None:
    """Per-lighting-condition aggregation — the thing MVTec AD 2 exists to measure."""
    from vlmab.eval.aggregate import aggregate
    from vlmab.eval.store import ResultStore

    df = ResultStore(results / "shards").load_all()
    print("\nPer lighting condition, pooled over the seeds present:")
    cols = ["meta_lighting", "i_auroc", "au_pro_030", "au_pro_005", "n"]
    table = aggregate(df, by="meta_lighting")
    print(table[[c for c in cols if c in table.columns]].to_string(index=False))
    print("\nNOTE: pooled across seeds for a first look only. Protocol §6 reports a mean over "
          "seeds with its standard deviation, and the metric functions refuse a pooled-seed "
          "frame — so this is a session convenience, never a reportable table.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--root", type=Path, default=Path("data/mvtec_ad2"))
    parser.add_argument("--results", type=Path, default=None,
                        help="default: results/mvtec_ad2/<category>")
    parser.add_argument("--archives", type=Path, default=DRIVE_ARCHIVES)
    parser.add_argument("--drive-results", type=Path, default=None,
                        help="default: <DRIVE_RESULTS>/<category>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-save", action="store_true",
                        help="skip the per-seed copy to Drive (not on Colab, say)")
    args = parser.parse_args(argv)

    results = args.results or Path("results/mvtec_ad2") / args.category
    drive_results = args.drive_results or DRIVE_RESULTS / args.category

    try:
        fetch(args.category, args.root, args.archives)
        verify_layout(args.category, args.root)
    except (FileNotFoundError, DriveNotMounted, RuntimeError,
            subprocess.CalledProcessError) as exc:
        print(f"\ncannot start: {exc}", file=sys.stderr)
        return 2

    if len(args.seeds) != len(set(args.seeds)):
        print(f"repeated seed in {args.seeds}; each seed is one run", file=sys.stderr)
        return 2
    if len(args.seeds) < 3:
        print(f"\n⚠️  {len(args.seeds)} seed(s). Protocol §6 requires three for any REPORTED "
              "number; anything less is a measurement. Continuing.\n")

    for i, seed in enumerate(args.seeds, 1):
        print(f"[seed {i}/{len(args.seeds)}] {args.category} at seed {seed}")
        run_one_seed(args.category, args.root, results, seed, args.device)
        if not args.no_save:
            save_to_drive(results, drive_results)

    summarise(results)
    print(f"\n{args.category} done at seeds {args.seeds}. Shards: {results / 'shards'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
