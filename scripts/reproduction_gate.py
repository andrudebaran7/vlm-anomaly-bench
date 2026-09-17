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

Protocol §6 runs three seeds, and the metric functions refuse a frame that pools them, so a
three-seed root needs one of two flags: `--seed N` scores that seed alone (`unseeded` selects
a run that applied none, which is not seed 0), and `--all-seeds` scores each separately and
reports the **mean of the per-seed means ± their sample std**. The std is reported and never
gates -- the pre-registered criterion in §2 is on the mean (§6 v0.2.13).

Exit codes: **0 PASS, 1 FAIL, 2 refused** — so a notebook cell or a CI job cannot print a
failing verdict and carry on, and a run that could not be scored at all is never mistaken for a
method that missed its target. A FAIL is a legitimate outcome, not a crash: the report is still
written. `--which secondary` scores the secondary target instead and never exits 1 — protocol §2
v0.2.12 fixes that it is reported with its caveat and cannot fail the method.
"""
import argparse
import datetime as dt
import statistics
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

from vlmab.eval.aggregate import image_metrics
from vlmab.eval.store import ResultStore, seed_tag

#: Our metrics are fractions in [0, 1]; the published tables are percentages. Every comparison
#: and everything printed is on the published scale, so the two are never mixed by eye.
PERCENT = 100.0


def _tags(df) -> "pd.Series":
    """Each row's seed as the readable tag the store already names shards by.

    Going through `seed_tag` rather than comparing the raw column keeps one definition of
    what a seed is called: `unseeded` is a real state (the run applied none) and is not the
    same as seed 0, and NaN is how parquet round-trips an unseeded run's None.
    """
    if "seed" not in df.columns:
        return pd.Series([seed_tag(None)] * len(df), index=df.index)
    return pd.Series(
        [seed_tag(None) if pd.isna(s) else seed_tag(int(s)) for s in df["seed"]],
        index=df.index,
    )


def _select_seed(df, want: str, results: Path):
    """Narrow a root that holds several seeds down to the one asked for.

    The metric functions refuse a pooled-seed frame (protocol §6), which is what makes a
    three-seed root unscorable by default. This is the deliberate narrowing that replaces it,
    and it names what is actually there when the asked-for seed is not.
    """
    wanted = seed_tag(None) if want == "unseeded" else seed_tag(int(want))
    tags = _tags(df)
    selected = df[tags == wanted]
    if selected.empty:
        raise ValueError(
            f"no rows under {results} have seed {wanted!r}; the root holds "
            f"{sorted(set(tags))}. Pass one of those to --seed, or drop --seed to score a "
            "single-seed root."
        )
    return selected


def _check_single_preprocess(df, results: Path) -> None:
    """Refuse a root whose shards were produced by more than one pre-processing.

    Nothing upstream stops this. The shard filename carries dataset/method/category/seed and
    no more, and `run_id` keys map directories on `config_hash`, so two pre-processings reach
    one root as soon as shards are copied between folders to be re-scored -- which is exactly
    how the 2026-09-16 run's shards were kept. Averaging them compares an adapter against a
    differently-configured version of itself and prints a verdict either way.
    """
    if "preprocess" not in df.columns:
        return
    found = sorted({str(v) for v in df["preprocess"].unique() if not pd.isna(v)})
    if len(found) > 1:
        raise ValueError(
            f"the shards under {results} carry {len(found)} different values of preprocess "
            f"({found}). Those are different configurations of the method, not repeats of one, "
            "so their mean is not a reproduction of anything. Score each from its own root."
        )


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
    target["n_seeds"] = cfg["n_seeds"]
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


def _per_seed_scores(df, target, results: Path) -> dict[str, dict[str, float]]:
    """Score each seed on its own: {seed tag: {category: I-AUROC%}}.

    Every seed is checked against the published category set individually. A seed that is
    missing a category cannot be averaged into the others -- its mean would be over a
    different set than theirs, and the combined number would be over neither.
    """
    per_seed = {}
    for tag, group in df.groupby(_tags(df), sort=True):
        scored = _scored(group, target, results)
        _check_categories(scored, target, results)
        per_seed[str(tag)] = scored
    expected = int(target["n_seeds"])
    if len(per_seed) != expected:
        raise ValueError(
            f"the root {results} holds {len(per_seed)} seeds ({sorted(per_seed)}), but "
            f"these targets pre-register {expected} seeds (protocol §6). A mean ± std over "
            "any other number is not the quantity the protocol asks for -- run the missing "
            "seeds, or score one alone with --seed."
        )
    return per_seed


def _across_seeds(per_seed: dict[str, dict[str, float]]) -> dict[str, tuple[float, float]]:
    """Per category, the mean across seeds and their sample std (ddof=1, the convention the
    published tables use). One seed has no spread to report, so its std is 0.0 rather than an
    error -- the seed COUNT is checked elsewhere, and this function should not have opinions
    about how many there should be."""
    categories = next(iter(per_seed.values())).keys()
    out = {}
    for cat in categories:
        values = [scored[cat] for scored in per_seed.values()]
        out[cat] = (
            statistics.fmean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0,
        )
    return out


def _report(scored, target, args, provenance, per_seed=None) -> str:
    # With several seeds the reported centre is the mean of the per-seed means, and the
    # spread is their sample std. Protocol §6 v0.2.13: the std is REPORTED, never gated --
    # the pre-registered criterion in §2 is on the mean, and a rule that also failed a band
    # reaching outside the tolerance would be a second criterion nobody registered.
    if per_seed:
        seed_means = {tag: sum(v.values()) / len(v) for tag, v in per_seed.items()}
        mean = statistics.fmean(seed_means.values())
        std = statistics.stdev(seed_means.values()) if len(seed_means) > 1 else 0.0
        measured = f"{mean:.2f} ± {std:.2f}"
    else:
        seed_means, std = {}, None
        mean = sum(scored.values()) / len(scored)
        measured = f"{mean:.2f}"
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
            f"Measured mean I-AUROC **{measured}** against published **{published:.1f}** "
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
            f"Measured mean I-AUROC **{measured}** against published **{published:.1f}** "
            f"(delta {delta:+.2f}).",
            "",
            "Protocol §2 v0.2.12: this number cannot fail the method. Caveat recorded with the "
            "target:",
            "",
            "> " + " ".join(str(target.get("caveat", "")).split()),
        ]

    lines += ["", f"Source: {target['source']}"]

    if per_seed:
        # Each seed's own mean, so a reader can see whether the spread comes from one
        # outlying run or from genuine run-to-run variance.
        lines += ["", "## Per seed", "", "| Seed | Mean I-AUROC |", "|---|---|"]
        for tag in sorted(seed_means):
            lines.append(f"| {tag} | {seed_means[tag]:.2f} |")

    lines += ["", "## Per category", ""]
    if target["per_category"]:
        spread = _across_seeds(per_seed) if per_seed else {}
        lines += ["| Category | Measured | Published | Delta |", "|---|---|---|---|"]
        for cat in sorted(scored):
            pub = float(target["per_category"][cat])
            d = scored[cat] - pub
            flag = " ⚠️" if abs(d) > tolerance else ""
            cell = (f"{scored[cat]:.2f} ± {spread[cat][1]:.2f}" if spread
                    else f"{scored[cat]:.2f}")
            lines.append(f"| {cat} | {cell} | {pub:.1f} | {d:+.2f}{flag} |")
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
    parser.add_argument(
        "--seed", default=None,
        help="score this seed alone out of a root that holds several (protocol §6 runs "
             "three). Takes an integer, or the literal `unseeded` for a run that applied "
             "no seed — which is not the same as seed 0.")
    parser.add_argument(
        "--all-seeds", action="store_true",
        help="score every seed under the root separately and report the mean of their means ± std (protocol §6). The std is reported, never gated.")
    args = parser.parse_args(argv)
    if args.all_seeds and args.seed is not None:
        raise ValueError(
            "--seed scores one seed and --all-seeds scores them all; passing both leaves "
            "the report titled as whichever lost. Pass one."
        )

    target = _load_targets(args.targets, args.which)

    df = ResultStore(args.results).load_all()
    if df.empty:
        raise ValueError(f"no shards under {args.results}")
    if args.seed is not None:
        df = _select_seed(df, args.seed, args.results)

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

    _check_single_preprocess(df, args.results)

    if args.all_seeds:
        per_seed = _per_seed_scores(df, target, args.results)
        scored = {cat: m for cat, (m, _) in _across_seeds(per_seed).items()}
    else:
        per_seed = None
        scored = _scored(df, target, args.results)
        _check_categories(scored, target, args.results)

    provenance = {"results_root": args.results, "targets": args.targets,
                  "n_categories": len(scored)}
    for column in ("commit", "seed", "gpu", "preprocess"):
        if column in df.columns:
            values = sorted({str(v) for v in df[column].unique()})
            provenance[column] = values[0] if len(values) == 1 else values

    text, passed = _report(scored, target, args, provenance, per_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")
    print(text.split("## Per category")[0])

    # A secondary check never gates, so it never reports failure through the exit code.
    return 0 if (passed or not target["gates"]) else 1


if __name__ == "__main__":
    # Exit codes: 0 PASS, 1 FAIL, 2 refused. A refusal is neither a verdict nor a crash -- the
    # run could not be scored at all -- so it must not be confused with FAIL, and in a notebook
    # a bare traceback buries the one line that says why.
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"refusing to score this run:\n\n  {exc}", file=sys.stderr)
        sys.exit(2)
