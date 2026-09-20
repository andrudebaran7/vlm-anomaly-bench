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
import datetime as dt
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


def restore_from_drive(results: Path, source: Path) -> None:
    """Copy back any shard Drive has and this VM does not, before deciding what to run.

    The runner resumes on `ResultStore.is_done`, which is an existence check on a shard file —
    and those live on the VM, which Colab reclaims. So a session that dropped after two of three
    seeds redid all three, because `save_to_drive` only ever copied outward. Found on the first
    live Vial run (2026-09-21), which re-fitted every seed on a fresh runtime while three
    perfectly good shards sat on Drive.

    **Only files missing locally are copied**, never overwriting a local shard with Drive's copy:
    the Drive copy came from the local one, so they agree, and a restore that can clobber is a
    restore nobody should run twice.

    The anomaly maps are NOT restored — they are not copied to Drive at all, being 24.9 GB across
    the grid against Drive's free 15 GB. A restored shard therefore supports image-level metrics
    and not pixel-level ones; `summarise` detects that and says so rather than dying on a missing
    file.
    """
    src = source / "shards"
    if not src.is_dir():
        return
    dest = results / "shards"
    dest.mkdir(parents=True, exist_ok=True)
    restored = [f for f in sorted(src.glob("*.parquet")) if not (dest / f.name).is_file()]
    for f in restored:
        shutil.copy2(f, dest / f.name)
    if restored:
        print(f"restored {len(restored)} shard(s) from {src}:")
        for f in restored:
            print(f"  {f.name}")
        print("  (their anomaly maps are not on Drive — see summarise's note if it skips "
              "pixel metrics)")


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


#: Metrics worth showing per lighting condition. MVTec AD 2 exists to measure the lighting shift,
#: so this view is the point of the whole grid rather than a convenience.
SUMMARY_METRICS = ("i_auroc", "au_pro_030", "au_pro_005")


def summarise(results: Path, max_bytes: int | None = None) -> None:
    """Per lighting condition, one aggregation PER SEED, combined as mean ± std (protocol §6).

    The first version of this pooled every seed into one `aggregate` call and died on the guard
    in `aggregate._require_single_seed` — after all three seeds had already been computed and
    saved. The guard was right and the caller was wrong: I-AUROC over 3N rows is not the mean of
    three I-AUROCs, it scores every image three times and treats the copies as independent.
    Metrics are computed once per seed and combined afterwards, which is what this now does.
    """
    import pandas as pd

    from vlmab.eval.aggregate import aggregate
    from vlmab.eval.store import ResultStore

    df = ResultStore(results / "shards").load_all()

    # A shard restored from Drive references maps that never left the VM it was computed on.
    # `pixel_metrics` loads every `map_path` from disk, so a missing file would raise from
    # inside np.load with no hint of why. Nulling the column instead makes `pixel_metrics`
    # return {"n": 0} — it already has that branch — so the image-level table still comes out.
    if "map_path" in df.columns:
        present = df["map_path"].map(lambda q: bool(q) and Path(q).is_file())
        missing = int((df["map_path"].notna() & ~present).sum())
        if missing:
            print(f"\n⚠️  {missing} of {int(df['map_path'].notna().sum())} anomaly maps are not "
                  "on this machine — shards restored from Drive carry paths to maps that stayed "
                  "on the VM that computed them.\n    Pixel metrics (AU-PRO, SegF1) are SKIPPED "
                  "for the affected rows; image-level metrics are unaffected.\n    To get them "
                  "back, delete the shard for that seed and re-run: the fit is minutes.")
            df = df.assign(map_path=df["map_path"].where(present, None))

    # NaN is how parquet round-trips an unseeded run's None; fold it back so it groups as one.
    tags = df["seed"].where(df["seed"].notna(), None) if "seed" in df.columns else None
    groups = list(df.groupby(tags, dropna=False)) if tags is not None else [(None, df)]

    kw = {} if max_bytes is None else {"max_bytes": max_bytes}
    per_seed = []
    for seed, group in groups:
        try:
            table = aggregate(group, by="meta_lighting", **kw)
        except ValueError as exc:
            if "max_bytes" not in str(exc):
                raise
            # The pixel-metric memory guard. It is a guard, not a tuning knob -- but its own
            # docstring says to raise it deliberately against measured free RAM, so the way out
            # is a measurement and an explicit flag, never a silent default. The shards are
            # already written and on Drive; only this summary is blocked.
            available = None
            try:
                for line in Path("/proc/meminfo").read_text().splitlines():
                    if line.startswith("MemAvailable:"):
                        available = int(line.split()[1]) * 1024
            except OSError:
                pass
            print(f"\n{exc}\n", file=sys.stderr)
            print("The shards are safe — this is the summary only, and nothing needs re-running.",
                  file=sys.stderr)
            if available:
                print(f"This machine reports {available / 1e9:.1f} GB available right now.",
                      file=sys.stderr)
            print("Re-run with an explicit budget you have checked against that number, e.g.:\n"
                  f"    python scripts/run_mvtec_ad2.py --category {results.name} "
                  "--max-bytes 9000000000\n", file=sys.stderr)
            raise SystemExit(1) from exc
        per_seed.append(table.assign(seed=seed))
    combined = pd.concat(per_seed, ignore_index=True)

    metrics = [m for m in SUMMARY_METRICS if m in combined.columns]
    stats = combined.groupby("meta_lighting")[metrics].agg(["mean", "std"])
    n_seeds = combined["seed"].nunique(dropna=False)

    def cell(metric, row):
        std = row[(metric, "std")]
        return f"{row[(metric, 'mean')]:.4f} ± {0.0 if pd.isna(std) else std:.4f}"

    print(f"\nPer lighting condition — mean ± std over {n_seeds} seed(s), "
          "each aggregated separately (protocol §6):")
    for lighting, row in stats.iterrows():
        print(f"  {lighting:<14} " + " | ".join(f"{m} {cell(m, row)}" for m in metrics))

    if n_seeds < 3:
        print(f"\nNOTE: {n_seeds} seed(s). Protocol §6 requires three for any reported number.")

    # A result that only ever printed is a result nobody has. `run_visa_secondary.py` writes its
    # report through the gate; this had no equivalent, and the first live M3 run's numbers
    # existed only in a console until they were transcribed by hand (2026-09-21).
    report = results.parent / f"{results.name}.md"
    seeds = sorted(str(s) for s in combined["seed"].dropna().unique()) or ["unseeded"]
    head = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent.parent),
                           "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip() or "unknown"
    lines = [
        f"# MVTec AD 2 — patchcore_ref on {results.name}",
        "",
        f"Generated {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC by `scripts/run_mvtec_ad2.py`.",
        "",
        f"Mean ± std over **{n_seeds} seed(s)** ({', '.join(seeds)}), each aggregated separately "
        "(protocol §6). The public test split; `test_private` has no ground truth here.",
        "",
        "| lighting | " + " | ".join(metrics) + " |",
        "|---|" + "---|" * len(metrics),
    ]
    lines += [f"| {lighting} | " + " | ".join(cell(m, row) for m in metrics) + " |"
              for lighting, row in stats.iterrows()]
    lines += [
        "",
        "## Caveats recorded with the result",
        "",
        "- **PatchCore is the full-shot anchor, not a zero-shot method.** It is the ceiling the "
        "zero-shot numbers are measured against, never a comparable entry.",
        "- **One category is not a dataset result.** MVTec AD 2 has eight.",
        "- **The pre-processor is a fixed square `Resize([256, 256])`** with no aspect-ratio "
        "preservation, and no MVTec AD 2 category is square. This is the pinned configuration "
        "the reproduction gate was passed at (protocol §7); it is a stated property of the "
        "study, not an explanation produced afterwards.",
        "",
        "## Provenance",
        "",
        f"- shards: `{results / 'shards'}`",
        f"- seeds: `{', '.join(seeds)}`",
        f"- commit: `{head}`",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print(f"\nwrote {report}")


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
    parser.add_argument("--summarise-only", action="store_true",
                        help="skip the seed loop entirely and only re-read the shards. This "
                             "never loads torch or anomalib, which is worth ~3 GB of resident "
                             "memory — the difference between a pixel-metric aggregation that "
                             "fits and one the kernel OOM-kills.")
    parser.add_argument("--max-bytes", type=int, default=None,
                        help="pixel-metric memory budget, in bytes. Only set this against a "
                             "number you have actually measured on the machine (the failure "
                             "message prints it); the default is a guard, not a knob.")
    parser.add_argument("--no-save", action="store_true",
                        help="skip the per-seed copy to Drive (not on Colab, say)")
    args = parser.parse_args(argv)

    # Printed FIRST, before anything can fail. A stale clone is invisible in a traceback --
    # the line numbers look plausible and the code is simply the old code. One live M3 run was
    # lost to exactly that (2026-09-21): a fix was on origin/master, the session had not synced,
    # and the traceback named a call signature that no longer existed anywhere in the repo.
    here = Path(__file__).resolve().parent.parent
    head = subprocess.run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip() or "unknown"
    print(f"{Path(__file__).name} running from commit {head}\n")

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

    if not args.no_save:
        restore_from_drive(results, drive_results)

    if args.summarise_only:
        print("--summarise-only: not loading torch or anomalib, and running no seed.\n")
        summarise(results, args.max_bytes)
        return 0

    for i, seed in enumerate(args.seeds, 1):
        print(f"[seed {i}/{len(args.seeds)}] {args.category} at seed {seed}")
        run_one_seed(args.category, args.root, results, seed, args.device)
        if not args.no_save:
            save_to_drive(results, drive_results)

    summarise(results, args.max_bytes)
    print(f"\n{args.category} done at seeds {args.seeds}. Shards: {results / 'shards'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
