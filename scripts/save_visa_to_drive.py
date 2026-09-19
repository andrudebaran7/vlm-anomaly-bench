#!/usr/bin/env python3
"""Copy whatever the VisA run has finished so far to Drive. Safe to run at any point.

    python scripts/save_visa_to_drive.py          # shards, plus the report if it exists

**Why this exists, and why it is a script rather than a notebook cell.** The VisA run's shards
live on the Colab VM's disk, and the VM's disk dies with the runtime. Cell 3.5 copies them, but
it runs *after* cell 3.4 finishes, and Colab serializes cells — so during a multi-hour run there
is no way to protect the objects already done. A dropped runtime at 11/12 costs everything.

The runner makes the recovery cheap on its side: a shard is written only after a category's whole
sample loop finishes, and `ResultStore.is_done` is an existence check on that file. So an
interrupted run costs the object in flight and nothing else, and re-running
`scripts/run_visa_secondary.py` skips every object whose shard is already there. The sequence
during a long run is therefore: watch for a `[N/12] <object>: done` line, interrupt, sync, run
this, re-run 3.4.

It is a tracked `.py` because a fix has to be able to reach a live session, and cell 1.1's hard
reset is how anything reaches one. It takes no arguments on purpose: one typed line, at a point
in a session where typing a long one is the last thing anybody wants.

Unlike cell 3.5 this does NOT require the report to exist — mid-run it does not, and refusing
would defeat the purpose. It says what it found either way.
"""
import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence

#: Deliberately not `reproduction/` or `reproduction_3seed/`, which hold MVTec AD shards under
#: filenames that carry no dataset name. The gate refuses a root whose shards disagree, but only
#: after someone has already merged two runs into one folder.
DEST = Path("/content/drive/MyDrive/reproduction_visa")
SHARDS = Path("results/reproduction/visa/shards")
REPORT = Path("results/reproduction/patchcore_visa.md")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=Path, default=SHARDS)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--no-mount", action="store_true",
                        help="Drive is already mounted (or this is not Colab)")
    args = parser.parse_args(argv)

    if not args.no_mount:
        try:
            from google.colab import drive       # noqa: PLC0415 — Colab-only, imported on use
            drive.mount("/content/drive")        # idempotent
        except ImportError:
            print("not on Colab; assuming --dest is already writable")

    if not args.shards.is_dir():
        print(f"nothing to copy: {args.shards} does not exist. Has the run started?",
              file=sys.stderr)
        return 1

    finished = sorted(p.name for p in args.shards.glob("*.parquet"))
    if not finished:
        print(f"nothing to copy: no shard has been written yet under {args.shards}. The first "
              "one appears only when the first object finishes.", file=sys.stderr)
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.shards, args.dest / "shards", dirs_exist_ok=True)
    print(f"copied {len(finished)} shard(s) to {args.dest / 'shards'}:")
    for name in finished:
        print(f"  {name}")

    # The maps are NOT copied and are not needed: the secondary check is image-level, and
    # threshold calibration runs on MVTec AD 2, not on VisA.
    if args.report.is_file():
        shutil.copy2(args.report, args.dest / args.report.name)
        print(f"copied {args.report.name} as well — the run had already scored itself")
    else:
        print(f"{args.report} does not exist yet, which is expected mid-run: the gate runs "
              "after the last object. The shards above are what makes it re-scorable.")

    print(f"\n{12 - len(finished)} of 12 objects still to go." if len(finished) < 12
          else "\nall 12 objects are safe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
