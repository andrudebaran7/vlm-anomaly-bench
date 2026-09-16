#!/usr/bin/env python3
"""Score a finished reproduction run against its pre-registered targets (protocol §2 v0.2.12).

This never runs a method. Running is GPU work; this is the CPU half — read the shards, compute
I-AUROC per category, compare against `configs/reproduction/<method>.yaml`, write the verdict.
The split is what makes the pass/fail logic testable at all, and it means a disputed verdict can
be recomputed from the shards without a GPU.

    # in Colab, once per category
    run_eval.py --method patchcore_ref --dataset mvtec_ad --root data/mvtec_ad \\
        --results runs/repro --split test --category bottle
    ...
    # anywhere
    reproduction_gate.py --results runs/repro \\
        --targets configs/reproduction/patchcore_ref.yaml \\
        --out results/reproduction/patchcore_mvtec_ad.md

Exit code is 0 on PASS and 1 on FAIL, so a notebook cell or a CI job cannot print a failing
verdict and carry on. A FAIL is a legitimate outcome, not a crash: the report is still written.
`--which secondary` scores the secondary target instead, and always exits 0 — protocol §2
v0.2.12 fixes that it is reported with its caveat and cannot fail the method.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Sequence

import yaml

from vlmab.eval.aggregate import image_metrics
from vlmab.eval.store import ResultStore

#: Our metrics are fractions in [0, 1]; the published tables are percentages. Every comparison
#: and everything printed is on the published scale, so the two are never mixed by eye.
PERCENT = 100.0


def _load_targets(path: Path, which: str) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if which == "gate":
        target = dict(cfg["gate"])
        target["per_category"] = cfg.get("published_per_category", {})
        target["n_categories"] = cfg["gate_n_categories"]
        target["gates"] = True
    else:
        target = dict(cfg["secondary"])
        target["per_category"] = {}
        target["n_categories"] = cfg["secondary_n_categories"]
        target["gates"] = False
    target["method"] = cfg["method"]
    return target


def _scored(df, target, results: Path) -> dict[str, float]:
    """I-AUROC per category, on the published percentage scale.

    `image_metrics` carries the guards this needs and they are not duplicated here: it refuses
    unlabelled rows, a single-class group, and a frame that pools seeds.
    """
    out = {}
    for category, group in df.groupby("category", sort=True):
        out[str(category)] = image_metrics(group)["i_auroc"] * PERCENT
    return out


def _check_categories(scored: dict[str, float], target: dict, results: Path) -> None:
    """Refuse a run whose categories are not exactly the published set.

    A mean over a subset is not the number the paper printed: it would pass or fail on which
    categories happened to finish, which is the opposite of a pre-registered criterion.
    """
    published = set(target["per_category"])
    found = set(scored)
    if published:
        missing = sorted(published - found)
        unexpected = sorted(found - published)
        if missing or unexpected:
            raise ValueError(
                f"the run under {results} does not cover the published category set: "
                f"missing {missing}, unexpected {unexpected}. The published mean is over "
                f"{target['n_categories']} categories, so a mean over any other set is not "
                "comparable to it (protocol §2)."
            )
    elif len(found) != target["n_categories"]:
        # The secondary target publishes no per-category breakdown, so only the count can be
        # checked — but an incomplete run is just as incomparable.
        raise ValueError(
            f"the run under {results} covers {len(found)} categories; the published number "
            f"averages {target['n_categories']}. Found: {sorted(found)}"
        )


def _report(scored, target, args, provenance) -> str:
    mean = sum(scored.values()) / len(scored)
    published = float(target["published_mean"])
    delta = mean - published
    tolerance = float(target.get("tolerance", 1.0))
    passed = abs(delta) <= tolerance

    lines = [
        f"# Reproduction {'gate' if target['gates'] else 'secondary check'} — "
        f"{target['method']} on {target['dataset']}",
        "",
        f"Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
        "`scripts/reproduction_gate.py`.",
        "",
    ]

    # Categories whose own delta exceeds the tolerance. The gate is on the mean (protocol §2),
    # so these do not change the verdict -- but a mean that passes while one category is seven
    # points low is a different situation from one that passes evenly, and a reader who stops
    # at the verdict would never see the difference. Naming them is the whole reason the
    # per-category targets are pre-registered alongside the mean.
    outliers = sorted(
        (cat, value - float(target["per_category"][cat]))
        for cat, value in scored.items()
        if cat in target["per_category"]
        and abs(value - float(target["per_category"][cat])) > tolerance
    )

    if target["gates"]:
        verdict = "PASS" if passed else "FAIL"
        lines += [
            f"## Verdict: **{verdict}**",
            "",
            f"Measured mean I-AUROC **{mean:.2f}** against published **{published:.1f}** "
            f"(delta {delta:+.2f}, tolerance ±{tolerance:.1f}).",
        ]
        if outliers:
            named = ", ".join(f"{c} ({d:+.2f})" for c, d in outliers)
            lines += [
                "",
                f"⚠️ **{len(outliers)} of {len(scored)} categories are outside ±{tolerance:.1f} "
                f"on their own:** {named}. The gate is on the mean, so this does not change the "
                "verdict — but it is not an even reproduction, and the cause belongs in the "
                "write-up before any frontier number is reported.",
            ]
    else:
        lines += [
            "## Secondary check — reported only, **does not gate**",
            "",
            f"Measured mean I-AUROC **{mean:.2f}** against published **{published:.1f}** "
            f"(delta {delta:+.2f}).",
            "",
            "Protocol §2 v0.2.12: this number cannot fail the method. Caveat recorded with the "
            "target:",
            "",
            "> " + " ".join(str(target.get("caveat", "")).split()),
        ]

    lines += ["", f"Source: {target['source']}", "", "## Per category", ""]
    if target["per_category"]:
        lines += ["| Category | Measured | Published | Delta |", "|---|---|---|---|"]
        for cat in sorted(scored):
            pub = float(target["per_category"][cat])
            d = scored[cat] - pub
            flag = " ⚠️" if abs(d) > tolerance else ""
            lines.append(f"| {cat} | {scored[cat]:.2f} | {pub:.1f} | {d:+.2f}{flag} |")
    else:
        lines += ["No per-category breakdown is published for this target, so the measured "
                  "values stand alone.", "", "| Category | Measured |", "|---|---|"]
        for cat in sorted(scored):
            lines.append(f"| {cat} | {scored[cat]:.2f} |")

    lines += ["", "## Provenance", ""]
    for key, value in provenance.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines), passed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="ResultStore root of the run")
    parser.add_argument("--targets", required=True, type=Path,
                        help="configs/reproduction/<method>.yaml")
    parser.add_argument("--out", required=True, type=Path, help="markdown report to write")
    parser.add_argument("--which", default="gate", choices=("gate", "secondary"))
    args = parser.parse_args(argv)

    target = _load_targets(args.targets, args.which)

    df = ResultStore(args.results).load_all()
    if df.empty:
        raise ValueError(f"no shards under {args.results}")

    # Check the shards' own columns rather than the caller's word: the filename does not carry
    # the dataset's identity in a way that scoring can rely on, and scoring one benchmark's run
    # against another's published numbers would still print a verdict.
    for column, expected in (("dataset", target["dataset"]), ("method", target["method"])):
        found = sorted({str(v) for v in df[column].unique()})
        if found != [expected]:
            raise ValueError(
                f"the shards under {args.results} have {column}={found}, but these targets are "
                f"for {column}={expected!r}. Point --results at the right run, or --targets at "
                "the right file."
            )

    scored = _scored(df, target, args.results)
    _check_categories(scored, target, args.results)

    provenance = {"results_root": args.results, "targets": args.targets,
                  "n_categories": len(scored)}
    for column in ("commit", "seed", "gpu"):
        if column in df.columns:
            values = sorted({str(v) for v in df[column].unique()})
            provenance[column] = values[0] if len(values) == 1 else values

    text, passed = _report(scored, target, args, provenance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")
    print(text.split("## Per category")[0])

    # A secondary check never gates, so it never reports failure through the exit code.
    return 0 if (passed or not target["gates"]) else 1


if __name__ == "__main__":
    sys.exit(main())
