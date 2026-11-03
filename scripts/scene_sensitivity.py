#!/usr/bin/env python3
"""How far the paired change aliased -> antialiased rests on single test scenes.

Protocol §6 v0.2.27, fixed before anything here was computed. Sheet Metal has 4 good scenes, and
the bootstrap of v0.2.26 cannot say whether one of them carries its change. For the two categories
whose change is free of the CUDA build (experiment A), this reads the shards phase 10 read (CPU
only) and reports, with no resampling:

* the v0.2.26 statistic and the paired change with each (folder, index) cluster removed in turn;
* each scene's share: the fraction of opposite-class images it ranks correctly (ties count half),
  averaged over seeds and conditions, per configuration; each class's shares average to I-AUROC;
* mean raw image score per class and configuration.

Reading rules, fixed in the protocol:

* the change is **scene-dependent** if removing any one scene flips its sign or leaves it below
  half the full change in size; otherwise it does not depend on any single scene;
* the hypothesis that the aliased forward inflates normal images' scores is **consistent** with the
  numbers if the correction lowers the mean good score, and by more than it lowers the mean bad one.

Both are descriptive: they say nothing about two scenes at once and re-decide no criterion.

    python scripts/scene_sensitivity.py \\
        --antialiased /content/drive/MyDrive/mvtec_ad2_results/patchcore_ref_v0.2.21 \\
        --aliased /content/drive/MyDrive/mvtec_ad2_results
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_iauroc import REPORTS, Paired, check_against_report, load  # noqa: E402

CATEGORIES = ("sheet_metal", "fabric")


def leave_one_out(p: Paired) -> list[dict]:
    """The statistic under each configuration, and the change, without each scene in turn."""
    rows = []
    for folder in sorted(set(p.folder)):
        for index in sorted(set(p.index[p.folder == folder])):
            kept = np.flatnonzero(~((p.folder == folder) & (p.index == index)))
            anti = p.stat(kept, "antialiased")[0]
            alias = p.stat(kept, "aliased")[0]
            rows.append({"folder": folder, "index": index, "antialiased": anti, "aliased": alias,
                         "change": anti - alias})
    return rows


def scene_shares(p: Paired, config: str) -> list[dict]:
    """Per scene: mean over seeds and conditions of the fraction of opposite-class images ranked
    correctly (ties half), and of its raw image score."""
    scores = p.scores[config]
    wins = np.empty(scores.shape)
    for k in range(len(p.conditions)):
        in_condition = np.flatnonzero(p.condition == k)
        for row in in_condition:
            others = in_condition[p.label[in_condition] != p.label[row]]
            sign = 1.0 if p.label[row] == 1 else -1.0
            diff = sign * (scores[row] - scores[others])    # others x seeds
            wins[row] = ((diff > 0) + 0.5 * (diff == 0)).mean(axis=0)
    out = []
    for folder in sorted(set(p.folder)):
        for index in sorted(set(p.index[p.folder == folder])):
            rows = np.flatnonzero((p.folder == folder) & (p.index == index))
            # Mean over conditions, as the statistic: a scene absent from one would not weigh less.
            share = np.mean([wins[rows[p.condition[rows] == k]].mean()
                             for k in sorted(set(p.condition[rows]))])
            out.append({"folder": folder, "index": index, "share": float(share),
                        "score": float(scores[rows].mean())})
    return out


def class_means(p: Paired, config: str) -> dict[str, float]:
    """Mean raw image score of each folder, over its images, conditions and seeds."""
    return {f: float(p.scores[config][p.folder == f].mean()) for f in sorted(set(p.folder))}


def scene_dependent(full: float, changes: Sequence[float]) -> bool:
    """The protocol's rule: any single removal flips the sign or leaves under half the size."""
    return any(c * full <= 0 or abs(c) < abs(full) / 2 for c in changes)


def normal_scores_inflated(delta_good: float, delta_bad: float) -> bool:
    """The correction (antialiased minus aliased) lowers good scores, and by more than bad ones."""
    return delta_good < 0 and delta_good < delta_bad


def section(category: str, p: Paired) -> list[str]:
    full = {c: p.stat(np.arange(len(p)), c)[0] for c in ("antialiased", "aliased")}
    change = full["antialiased"] - full["aliased"]
    loo = leave_one_out(p)
    changes = [r["change"] for r in loo]
    dependent = scene_dependent(change, changes)
    shares = {c: {(s["folder"], s["index"]): s for s in scene_shares(p, c)} for c in full}
    means = {c: class_means(p, c) for c in full}
    delta = {f: means["antialiased"][f] - means["aliased"][f] for f in ("good", "bad")}
    print(f"{category}: change {change:+.3f}; leave-one-out {min(changes):+.3f} to "
          f"{max(changes):+.3f}; {'scene-dependent' if dependent else 'no single scene decides'}")
    lines = [f"## {category}", "",
             f"Aliased {full['aliased']:.4f}, antialiased {full['antialiased']:.4f}, change "
             f"{change:+.4f}. Removing one scene: change {min(changes):+.4f} to {max(changes):+.4f}. "
             f"**Rule (half the size is {abs(change) / 2:.4f}): "
             f"{'scene-dependent' if dependent else 'does not depend on any single scene'}.**", "",
             "| Scene left out | Aliased | Antialiased | Change | Scene's share, aliased | "
             "Scene's share, antialiased | Mean score, aliased | Mean score, antialiased |",
             "|---|---|---|---|---|---|---|---|"]
    for r in loo:
        key = (r["folder"], r["index"])
        a, b = shares["aliased"][key], shares["antialiased"][key]
        lines.append(f"| {r['folder']}/{r['index']} | {r['aliased']:.4f} | {r['antialiased']:.4f} | "
                     f"{r['change']:+.4f} | {a['share']:.4f} | {b['share']:.4f} | "
                     f"{a['score']:.3f} | {b['score']:.3f} |")
    inflated = normal_scores_inflated(delta["good"], delta["bad"])
    lines += ["", "| Class | Mean score, aliased | Mean score, antialiased | Correction |",
              "|---|---|---|---|"]
    lines += [f"| {f} | {means['aliased'][f]:.3f} | {means['antialiased'][f]:.3f} | {delta[f]:+.3f} |"
              for f in ("good", "bad")]
    lines += ["", f"**Hypothesis that the aliased forward inflates normal scores: "
              f"{'consistent' if inflated else 'not consistent'} with these means.**", ""]
    return lines


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs the shards
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--antialiased", type=Path, required=True)
    parser.add_argument("--aliased", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--out", type=Path, default=Path("results/bootstrap/scene_sensitivity.md"))
    args = parser.parse_args(argv)

    head = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse",
                           "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"{Path(__file__).name} running from commit {head or 'unknown'}\n")
    lines = ["# Sensitivity of the paired I-AUROC change to single scenes, MVTec AD 2 test_public", "",
             f"Generated by `scripts/scene_sensitivity.py` at commit `{head or 'unknown'}`; protocol "
             "§6 v0.2.27. Descriptive; no resampling. A scene's share is the fraction of "
             "opposite-class images it ranks correctly, ties half, over seeds and conditions.", ""]
    roots = {"antialiased": args.antialiased, "aliased": args.aliased}
    for category in args.categories:
        tables = []
        for config, root in roots.items():
            table = load(root, category, config)
            bad = check_against_report(Paired([table]), config,
                                       (REPORTS[config] / f"{category}.md").read_text())
            if bad:
                print(f"REFUSED {category}/{config}: shards do not reproduce the report: {bad}")
                return 2
            tables.append(table)
        lines += section(category, Paired(tables))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
