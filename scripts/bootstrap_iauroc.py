#!/usr/bin/env python3
"""Paired, scene-clustered bootstrap of the anchor's I-AUROC on MVTec AD 2 (protocol v0.2.26).

Both external reviews of 2026-11-02 asked for the uncertainty the three seeds do not measure:
how much I-AUROC would move with a different draw of test images. This reads the committed grids'
shards (CPU only) and reports 95% percentile intervals for

* each category's I-AUROC under the corrected configuration (v0.2.21), per condition and as the
  category mean the paper's Table II prints, plus the means over all eight and over the five usable;
* the paired change aliased -> antialiased per category, both configurations scored on the SAME
  resampled images in every replicate.

The scheme, fixed in protocol v0.2.26 before any interval was computed:

* **Resampling unit: the scene index.** A test image is `{good|bad}/{index}_{condition}.png`; all
  images sharing (folder, index) form one cluster and are drawn together, across every lighting
  condition, so conditions that may show one scene are never treated as independent.
* **Stratified by folder** (good / bad): each replicate draws as many good and bad clusters as the
  category has, with replacement, so every replicate has both classes.
* **Seeds are not resampled.** Each replicate scores all three seeds on the drawn images and
  averages them, exactly as the reported figure does: seed spread is reported separately.
* **Statistic:** per condition, the mean over seeds of I-AUROC; per category, the unweighted mean
  over conditions of that. A replicate in which some condition lacks a class is redrawn, and the
  number of redraws is printed.
* B = 10,000 replicates, numpy generator seeded with 20261102, 2.5th and 97.5th percentiles.

Before bootstrapping, each configuration's shards must reproduce its committed report's
per-condition I-AUROC at the report's own decimals; a category that does not is refused.

    python scripts/bootstrap_iauroc.py \\
        --antialiased /content/drive/MyDrive/mvtec_ad2_results/patchcore_ref_v0.2.21 \\
        --aliased /content/drive/MyDrive/mvtec_ad2_results

The aliased root is the un-scoped one: those shards predate the method-scoped layout of 2026-10-01
and were never migrated (run_mvtec_ad2.default_results). Fruit Jelly's are absent from Drive, so it
gets the anchor's interval only.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

CATEGORIES = ("vial", "walnuts", "sheet_metal", "fruit_jelly", "fabric", "rice", "wallplugs",
              "can")
USABLE = ("vial", "walnuts", "sheet_metal", "fruit_jelly", "fabric")
#: Experiment A showed the build moved nothing on these two; elsewhere aliased -> antialiased
#: also changes the CUDA build (cu128 -> cu130), and the change is reported with that caveat.
BUILD_ISOLATED = ("sheet_metal", "fabric")
REPORTS = {"antialiased": Path("results/mvtec_ad2/patchcore_ref_v0.2.21"),
           "aliased": Path("results/mvtec_ad2/patchcore_ref")}
B = 10_000
RNG_SEED = 20261102


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties -- equal to sklearn's roc_auc_score."""
    from scipy.stats import rankdata

    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def image_table(shards: pd.DataFrame, config: str) -> pd.DataFrame:
    """One row per test_public image and seed: folder, index, condition, label, score."""
    df = shards[(shards["split"] == "test_public") & shards["label"].isin([0, 1])]
    paths = df["image_path"].map(Path)
    out = pd.DataFrame({
        "folder": paths.map(lambda p: p.parent.name).to_numpy(),
        "index": paths.map(lambda p: p.stem.partition("_")[0]).to_numpy(),
        "condition": df["meta_lighting"].to_numpy(),
        "label": df["label"].astype(int).to_numpy(),
        "seed": df["seed"].astype(str).to_numpy(),
        "score": df["image_score"].astype(float).to_numpy(),
    })
    if out.duplicated(["folder", "index", "condition", "seed"]).any():
        raise ValueError(f"{config}: an image appears twice for one seed")
    return out.assign(config=config)


class Paired:
    """The paired test set of one category as arrays: one row per image, one score column per
    seed for each configuration. Built from the long tables; an image missing from any
    configuration or seed is refused rather than dropped, so every side sees one test set."""

    def __init__(self, tables: Sequence[pd.DataFrame]):
        long = pd.concat(tables, ignore_index=True)
        keys = ["folder", "index", "condition", "label"]
        w = long.pivot_table(index=keys, columns=["config", "seed"], values="score",
                             aggfunc="first")
        if w.isna().any().any():
            raise ValueError("an image is scored under one configuration or seed but not another")
        frame = w.index.to_frame(index=False)
        self.folder = frame["folder"].to_numpy()
        self.index = frame["index"].to_numpy()
        self.label = frame["label"].to_numpy().astype(int)
        self.conditions = sorted(frame["condition"].unique())
        self.condition = frame["condition"].map(self.conditions.index).to_numpy()
        self.scores = {c: w[c].to_numpy(dtype=np.float64) for c in w.columns.get_level_values(0).unique()}

    def __len__(self) -> int:
        return len(self.label)

    def stat(self, rows: np.ndarray, config: str):
        """Mean over conditions of the mean over seeds of I-AUROC on `rows` (repeats allowed).
        Returns (category value, {condition: value}), or (None, {}) if a condition lacks a class."""
        per_condition = {}
        for k, name in enumerate(self.conditions):
            r = rows[self.condition[rows] == k]
            labels = self.label[r]
            if len(r) == 0 or labels.min() == labels.max():
                return None, {}
            s = self.scores[config][r]
            per_condition[name] = float(np.mean([auroc(labels, s[:, j]) for j in range(s.shape[1])]))
        return float(np.mean(list(per_condition.values()))), per_condition

    def strata(self) -> list[list[np.ndarray]]:
        """Row positions of each (folder, index) cluster, one list per folder (the strata)."""
        out: dict[str, list[np.ndarray]] = {}
        for folder in sorted(set(self.folder)):
            in_folder = np.flatnonzero(self.folder == folder)
            for idx in sorted(set(self.index[in_folder])):
                out.setdefault(folder, []).append(in_folder[self.index[in_folder] == idx])
        return list(out.values())


def bootstrap(p: "Paired", configs: Sequence[str], b: int, rng: np.random.Generator):
    """b paired replicates of each configuration's category value, and the redraw count."""
    strata = p.strata()
    values = {c: np.empty(b) for c in configs}
    per_condition = {c: {name: np.empty(b) for name in p.conditions} for c in configs}
    redraws, i = 0, 0
    while i < b:
        rows = np.concatenate([members[j] for members in strata
                               for j in rng.integers(0, len(members), len(members))])
        results = {c: p.stat(rows, c) for c in configs}
        if any(v is None for v, _ in results.values()):
            redraws += 1
            continue
        for c, (v, conds) in results.items():
            values[c][i] = v
            for name, x in conds.items():
                per_condition[c][name][i] = x
        i += 1
    return values, per_condition, redraws


def interval(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def check_against_report(p: "Paired", config: str, report_text: str) -> list[str]:
    """The shards' per-condition seed-mean I-AUROC at the report's decimals; mismatches named."""
    from vlmab.viz.tables import parse_report

    _, conds = p.stat(np.arange(len(p)), config)
    reported = {r.condition: str(r.mean["i_auroc"]) for r in parse_report(report_text)}
    bad = []
    for condition, want in reported.items():
        places = len(want.split(".")[1])
        got = f"{conds.get(condition, float('nan')):.{places}f}"
        if got != want:
            bad.append(f"{condition}: shards {got}, report {want}")
    return bad


def load(root: Path, category: str, config: str) -> pd.DataFrame:  # pragma: no cover - Drive
    from vlmab.eval.store import ResultStore

    shards = root / category / "shards"
    if not shards.is_dir() or not any(shards.glob("*.parquet")):
        raise FileNotFoundError(f"{config}: no {shards}")
    return image_table(ResultStore(shards).load_all(), config)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs the shards
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--antialiased", type=Path, required=True,
                        help="root holding <category>/shards of the v0.2.21 grid")
    parser.add_argument("--aliased", type=Path, default=None,
                        help="root holding <category>/shards of the pre-v0.2.21 grid; "
                             "without it only the anchor's intervals are computed")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--b", type=int, default=B)
    parser.add_argument("--out", type=Path, default=Path("results/bootstrap/iauroc_bootstrap.md"))
    args = parser.parse_args(argv)

    head = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse",
                           "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"{Path(__file__).name} running from commit {head or 'unknown'}; B={args.b}, "
          f"rng seed {RNG_SEED}\n")
    rng = np.random.default_rng(RNG_SEED)
    roots = {"antialiased": args.antialiased}
    if args.aliased is not None:
        roots["aliased"] = args.aliased

    lines = ["# Paired scene-clustered bootstrap of I-AUROC, MVTec AD 2 test_public",
             "", f"Generated by `scripts/bootstrap_iauroc.py` at commit `{head or 'unknown'}`; "
             f"protocol v0.2.26. B = {args.b}, numpy seed {RNG_SEED}, 95% percentile intervals. "
             "Resampling unit: (folder, scene index) across all lighting conditions, stratified "
             "by folder; seeds averaged, not resampled.", ""]
    anchor_reps, rows = {}, []
    for category in args.categories:
        tables, refused = {}, False
        for config, root in roots.items():
            try:
                table = load(root, category, config)
            except FileNotFoundError as exc:
                if config == "antialiased":
                    raise
                print(f"{category}: {exc} -- anchor interval only, no paired change")
                continue
            bad = check_against_report(Paired([table]), config,
                                       (REPORTS[config] / f"{category}.md").read_text())
            if bad:
                print(f"REFUSED {category}/{config}: shards do not reproduce the report: {bad}")
                refused = True
                break
            tables[config] = table
        if refused:
            continue
        p = Paired(list(tables.values()))
        values, per_condition, redraws = bootstrap(p, list(tables), args.b, rng)
        point = {c: p.stat(np.arange(len(p)), c) for c in tables}
        anchor_reps[category] = values["antialiased"]
        n_good, n_bad = (len(set(p.index[p.folder == f])) for f in ("good", "bad"))
        row = {"category": category, "clusters": f"{n_good} good / {n_bad} bad",
               "redraws": redraws, "anchor": point["antialiased"][0],
               "anchor_ci": interval(values["antialiased"])}
        if "aliased" in tables:
            delta = values["antialiased"] - values["aliased"]
            row.update(aliased=point["aliased"][0],
                       delta=point["antialiased"][0] - point["aliased"][0],
                       delta_ci=interval(delta))
        row["conditions"] = {c: (point["antialiased"][1][c], interval(per_condition["antialiased"][c]))
                             for c in sorted(point["antialiased"][1])}
        rows.append(row)
        print(f"{category}: anchor {row['anchor']:.3f} [{row['anchor_ci'][0]:.3f}, "
              f"{row['anchor_ci'][1]:.3f}]" + (f"; change {row['delta']:+.3f} "
              f"[{row['delta_ci'][0]:+.3f}, {row['delta_ci'][1]:+.3f}]" if "delta" in row else "")
              + f"; {redraws} redraws")

    lines += ["## Anchor I-AUROC per category (v0.2.21)", "",
              "| Category | Clusters | I-AUROC | 95% interval | Redraws |", "|---|---|---|---|---|"]
    lines += [f"| {r['category']} | {r['clusters']} | {r['anchor']:.3f} | "
              f"[{r['anchor_ci'][0]:.3f}, {r['anchor_ci'][1]:.3f}] | {r['redraws']} |" for r in rows]
    for name, subset in (("all eight", CATEGORIES), ("the five usable", USABLE)):
        if all(c in anchor_reps for c in subset):
            reps = np.mean([anchor_reps[c] for c in subset], axis=0)
            pt = np.mean([r["anchor"] for r in rows if r["category"] in subset])
            lo, hi = interval(reps)
            lines.append(f"\nMean over {name}: {pt:.3f} [{lo:.3f}, {hi:.3f}] (categories drawn "
                         "independently, then averaged per replicate).")
    if any("delta" in r for r in rows):
        lines += ["", "## Paired change aliased -> antialiased per category", "",
                  "| Category | Aliased | Antialiased | Change | 95% interval | Build isolated |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {r['category']} | {r['aliased']:.3f} | {r['anchor']:.3f} | "
                  f"{r['delta']:+.3f} | [{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}] | "
                  f"{'yes (experiment A)' if r['category'] in BUILD_ISOLATED else 'no: cu128 vs cu130'} |"
                  for r in rows if "delta" in r]
        unpaired = [r["category"] for r in rows if "delta" not in r]
        if unpaired:
            lines += ["", f"No paired change for {', '.join(unpaired)}: aliased shards absent."]
    lines += ["", "## Anchor I-AUROC per lighting condition (v0.2.21)", "",
              "| Category | Condition | I-AUROC | 95% interval |", "|---|---|---|---|"]
    for r in rows:
        lines += [f"| {r['category']} | {c} | {v:.3f} | [{lo:.3f}, {hi:.3f}] |"
                  for c, (v, (lo, hi)) in r["conditions"].items()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
