"""The paper's anchor tables, generated from the committed MVTec AD 2 reports.

Input is the per-condition table each `results/mvtec_ad2/<label>/<category>.md` report carries
(mean ± std over seeds, protocol §6); output is LaTeX rows the paper `\\input`s. Two tables:

* Table 2 (`anchor_rows`): per category, the mean over lighting conditions of the per-condition
  seed mean, its range over conditions, and AU-PRO as a multiple of its random baseline.
* The per-condition table (`per_condition_rows`): every condition's mean ± std over seeds, verbatim
  from the reports at their four decimals. Table 2's range is spread over *conditions*; this is the
  spread over *seeds* that §6 requires to be reported.

Rounding is Decimal round-half-up, as in the tables first published on 2026-10-28; Python's
`round` would round half to even and move a few cells by one in the last place.
"""
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

METRICS = ("i_auroc", "au_pro_030", "au_pro_005")

#: AU-PRO of an uninformative map at the 0.3 and 0.05 limits: analytically L/2. Until 2026-10-31
#: this held the implementation's measured 0.1487 / 0.0249, which no longer reproduce (the test's
#: fixture gives 0.1503 / 0.0248; tests/test_au_pro_random_baseline.py pins only +-5 % of L/2), so
#: the paper divides by the analytic value.
RANDOM_BASELINE = {"au_pro_030": Decimal("0.15"), "au_pro_005": Decimal("0.025")}

#: The two categories whose verdict the v0.2.21 correction changed. Experiment A (2026-10-30/31,
#: docs/next-steps.md "RESULT") reproduced both reports of each, cell for cell, on one runtime, so
#: their before/after is the forward resize alone.
ABLATION_CATEGORIES = ("sheet_metal", "fabric")

#: Table 2's row order (by I-AUROC under v0.2.21) and the paper's category names.
CATEGORY_ORDER = ("vial", "walnuts", "sheet_metal", "fruit_jelly", "fabric", "rice", "wallplugs",
                  "can")
NAMES = {"vial": "Vial", "walnuts": "Walnuts", "sheet_metal": "Sheet Metal",
         "fruit_jelly": "Fruit Jelly", "fabric": "Fabric", "rice": "Rice",
         "wallplugs": "Wall Plugs", "can": "Can"}

#: Protocol §3 v0.2.22: the anchor is a usable ceiling where its detection is clearly above
#: chance. The rule fixes no number, so the outcome is recorded here rather than computed --
#: and it was decided after the v0.2.21 numbers existed (post hoc).
USABLE = frozenset({"vial", "walnuts", "sheet_metal", "fruit_jelly", "fabric"})

_ROW = re.compile(r"^\| (\w+) \| ([\d.]+) ± ([\d.]+) \| ([\d.]+) ± ([\d.]+) \| ([\d.]+) ± ([\d.]+) \|$")


@dataclass(frozen=True)
class ConditionRow:
    condition: str
    mean: dict
    std: dict


@dataclass(frozen=True)
class Summary:
    i_auroc: Decimal
    i_min: Decimal
    i_max: Decimal
    au_pro_030: Decimal
    au_pro_005: Decimal


def fmt(x: Decimal, places: int = 3) -> str:
    return str(x.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def parse_report(text: str) -> list[ConditionRow]:
    """The per-condition rows of one report, in the report's order."""
    rows = []
    for line in text.splitlines():
        m = _ROW.match(line.strip())
        if m:
            g = m.groups()
            rows.append(ConditionRow(
                g[0],
                {k: Decimal(g[1 + 2 * i]) for i, k in enumerate(METRICS)},
                {k: Decimal(g[2 + 2 * i]) for i, k in enumerate(METRICS)},
            ))
    if not rows:
        raise ValueError("no per-condition rows found; is this an MVTec AD 2 category report?")
    return rows


def summarise(rows: list[ConditionRow]) -> Summary:
    n = len(rows)
    mean = {k: sum(r.mean[k] for r in rows) / n for k in METRICS}
    i = [r.mean["i_auroc"] for r in rows]
    return Summary(mean["i_auroc"], min(i), max(i), mean["au_pro_030"], mean["au_pro_005"])


def _multiple(value: Decimal, metric: str) -> str:
    return f"{fmt(value)} ({fmt(value / RANDOM_BASELINE[metric], 2)}$\\times$)"


def anchor_rows(reports: dict[str, str]) -> list[str]:
    """Table 2's body: usable categories, a rule, then the rest."""
    rows = []
    for c in CATEGORY_ORDER:
        if c not in USABLE and not any(r == r"\midrule" for r in rows):
            rows.append(r"\midrule")
        s = summarise(parse_report(reports[c]))
        status = "usable" if c in USABLE else "not usable"
        rows.append(f"{NAMES[c]} & {fmt(s.i_auroc)} & {fmt(s.i_min)}--{fmt(s.i_max)} & "
                    f"{_multiple(s.au_pro_030, 'au_pro_030')} & "
                    f"{_multiple(s.au_pro_005, 'au_pro_005')} & {status} \\\\")
    return rows


def usable_mean(reports: dict[str, str]) -> tuple[str, str, str]:
    """Mean over the usable categories of each category's mean over conditions."""
    s = [summarise(parse_report(reports[c])) for c in CATEGORY_ORDER if c in USABLE]
    return tuple(fmt(sum(getattr(x, k) for x in s) / len(s)) for k in METRICS)


def per_condition_rows(reports: dict[str, str]) -> list[str]:
    """Every condition's seed mean ± std, verbatim at the reports' four decimals."""
    rows = []
    for c in CATEGORY_ORDER:
        if rows:
            rows.append(r"\midrule")
        for r in parse_report(reports[c]):
            cells = " & ".join(f"{r.mean[k]} $\\pm$ {r.std[k]}" for k in METRICS)
            rows.append(f"{NAMES[c]} & {r.condition.replace('_', chr(92) + '_')} & {cells} \\\\")
    return rows


def ablation_rows(aliased: dict[str, str], antialiased: dict[str, str]) -> list[str]:
    """The before/after table: each category's mean over conditions under both forward resizes."""
    rows = []
    for c in ABLATION_CATEGORIES:
        if rows:
            rows.append(r"\midrule")
        for label, reports in (("aliased", aliased), ("antialiased", antialiased)):
            s = summarise(parse_report(reports[c]))
            rows.append(f"{NAMES[c]} & {label} & {fmt(s.i_auroc)} & "
                        f"{_multiple(s.au_pro_030, 'au_pro_030')} & "
                        f"{_multiple(s.au_pro_005, 'au_pro_005')} \\\\")
    return rows
