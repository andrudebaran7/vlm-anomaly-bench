"""scripts/antialias_ablation.py: experiment A, both forward resizes on one VM (2026-10-30)."""
import pandas as pd
import pytest


def _table(seed, rows):
    return pd.DataFrame([{"meta_lighting": c, "i_auroc": i, "au_pro_030": a, "au_pro_005": b}
                         for c, i, a, b in rows]).assign(seed=seed)


def test_seed_stats_is_mean_and_sample_std_per_condition():
    from antialias_ablation import seed_stats

    stats = seed_stats([_table(0, [("regular", 0.5, 0.2, 0.05)]),
                        _table(1, [("regular", 0.7, 0.2, 0.05)])])
    row = stats.loc["regular"]
    assert row[("i_auroc", "mean")] == pytest.approx(0.6)
    assert row[("i_auroc", "std")] == pytest.approx(0.1414213, rel=1e-5)   # ddof=1, as summarise


def test_compare_formats_at_the_reports_own_precision():
    from antialias_ablation import compare, seed_stats

    report = ("| lighting | i_auroc | au_pro_030 | au_pro_005 |\n|---|---|---|---|\n"
              "| regular | 0.5500 ± 0.0000 | 0.2126 ± 0.0045 | 0.0570 ± 0.0015 |\n"
              "| shift_1 | 0.46 ± 0.01 | 0.29 ± 0.00 | 0.08 ± 0.00 |\n")
    stats = seed_stats([_table(0, [("regular", 0.55, 0.21264, 0.0570),
                                   ("shift_1", 0.4611, 0.2938, 0.0800)])])
    out = {(r["condition"], r["metric"]): r for r in compare(stats, report)}
    assert out[("regular", "i_auroc")]["equal"] is True
    assert out[("regular", "au_pro_030")]["measured"] == "0.2126"
    assert out[("shift_1", "i_auroc")]["measured"] == "0.46"          # two decimals, like the row
    assert out[("shift_1", "i_auroc")]["equal"] is True


def test_compare_names_a_condition_the_report_lacks():
    from antialias_ablation import compare, seed_stats

    report = "| regular | 0.5500 ± 0.0000 | 0.2126 ± 0.0045 | 0.0570 ± 0.0015 |\n"
    stats = seed_stats([_table(0, [("shift_9", 0.5, 0.2, 0.05)])])
    with pytest.raises(KeyError, match="shift_9"):
        compare(stats, report)


def test_the_two_modes_are_the_two_values_of_the_backend_switch():
    from antialias_ablation import MODES

    assert MODES == {"aliased": True, "antialiased": False}


def test_a_shard_that_does_not_record_its_mode_is_refused():
    from antialias_ablation import check_mode

    df = pd.DataFrame({"resize_antialias": [True, True]})
    check_mode(df, "antialiased")
    with pytest.raises(RuntimeError, match="aliased"):
        check_mode(df, "aliased")
    with pytest.raises(RuntimeError, match="resize_antialias"):
        check_mode(pd.DataFrame({"x": [1]}), "aliased")
