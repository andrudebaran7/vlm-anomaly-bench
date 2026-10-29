"""The paper's anchor tables are generated from the committed reports, never typed.

Table 2 was first produced by an uncommitted script (2026-10-28); the review of 2026-10-29 asked for
it in the repo, and for the per-condition seed spread protocol §6 requires to be reported. These
tests pin the parsing, the round-half-up rounding the paper uses, and both tables against the eight
committed v0.2.21 reports.
"""
from decimal import Decimal
from pathlib import Path

import pytest

from vlmab.viz import tables

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "results" / "mvtec_ad2" / "patchcore_ref_v0.2.21"

REPORT = """# MVTec AD 2 — patchcore_ref on toy

| lighting | i_auroc | au_pro_030 | au_pro_005 |
|---|---|---|---|
| regular | 0.8800 ± 0.0353 | 0.9033 ± 0.0004 | 0.6067 ± 0.0013 |
| shift_1 | 0.9375 ± 0.0154 | 0.2975 ± 0.0006 | 0.0125 ± 0.0013 |

## Caveats recorded with the result
| not | a | condition | row |
"""


def test_parse_reads_every_condition_row_and_nothing_else():
    rows = tables.parse_report(REPORT)
    assert [r.condition for r in rows] == ["regular", "shift_1"]
    assert rows[0].mean["i_auroc"] == Decimal("0.8800")
    assert rows[0].std["au_pro_005"] == Decimal("0.0013")


def test_parse_refuses_a_report_without_condition_rows():
    with pytest.raises(ValueError):
        tables.parse_report("# empty\n")


def test_rounding_is_half_up_not_bankers():
    # 0.9375 + 0.8800 over two = 0.90875 -> 0.909 half-up (banker's would give 0.908).
    assert tables.fmt(Decimal("0.90875")) == "0.909"
    assert tables.fmt(Decimal("0.0125"), 2) == "0.01"
    assert tables.fmt(Decimal("0.125"), 2) == "0.13"


def test_category_summary_is_the_mean_over_conditions():
    s = tables.summarise(tables.parse_report(REPORT))
    assert s.i_auroc == (Decimal("0.8800") + Decimal("0.9375")) / 2
    assert (s.i_min, s.i_max) == (Decimal("0.8800"), Decimal("0.9375"))


def _committed():
    return {c: (REPORTS / f"{c}.md").read_text() for c in tables.CATEGORY_ORDER}


def test_table2_rows_match_the_paper_as_published_2026_10_28():
    rows = tables.anchor_rows(_committed())
    assert rows[0] == (r"Vial & 0.888 & 0.853--0.938 & 0.863 (5.81$\times$) & "
                       r"0.567 (22.79$\times$) & usable \\")
    assert rows[5] == r"\midrule"
    assert rows[6] == (r"Rice & 0.537 & 0.460--0.619 & 0.433 (2.91$\times$) & "
                       r"0.173 (6.94$\times$) & not usable \\")
    assert rows[-1] == (r"Can & 0.452 & 0.370--0.515 & 0.237 (1.59$\times$) & "
                        r"0.015 (0.62$\times$) & not usable \\")
    assert len(rows) == 9


def test_usable_mean_matches_the_caption():
    assert tables.usable_mean(_committed()) == ("0.808", "0.545", "0.293")


def test_per_condition_table_has_one_row_per_condition_with_seed_std():
    rows = tables.per_condition_rows(_committed())
    body = [r for r in rows if "&" in r]
    assert len(body) == 6 * 6 + 7 + 4   # six categories with 6 conditions, Vial 7, Fruit Jelly 4
    # Four decimals, verbatim from the reports: the AU-PRO seed spread is often below 0.001, so a
    # three-decimal std would print 0.000 for a spread that exists.
    assert (r"Vial & underexposed & 0.8533 $\pm$ 0.0481 & 0.6479 $\pm$ 0.0077 & "
            r"0.3591 $\pm$ 0.0108 \\") in rows
