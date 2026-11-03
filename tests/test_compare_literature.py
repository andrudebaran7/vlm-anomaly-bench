"""scripts/compare_literature.py: the post hoc comparison the paper quotes (protocol v0.2.28)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_literature", ROOT / "scripts" / "compare_literature.py")
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)


def _reports():
    return {c: (cl.REPORTS / f"{c}.md").read_text() for c in cl.CATEGORIES}


def test_anchor_values_are_the_tables():
    ours = cl.anchor(_reports())
    i = cl.CATEGORIES.index("sheet_metal")
    assert cl.tables.fmt(ours["i_auroc"][i]) == "0.797"
    assert cl.tables.fmt(ours["au_pro_030"][i]) == "0.228"
    assert cl.tables.fmt(ours["au_pro_005"][cl.CATEGORIES.index("vial")]) == "0.567"
    assert cl.tables.fmt(ours["i_auroc"][cl.CATEGORIES.index("rice")]) == "0.537"


def test_literature_means_match_the_published_mean_rows():
    """The transcribed rows must average to the Mean row each table prints."""
    means = {("au_pro_005", "TEST_priv"): 28.8, ("au_pro_005", "TEST_priv,mix"): 26.0,
             ("au_pro_030", "TEST_priv"): 53.8, ("au_pro_030", "TEST_priv,mix"): 50.8}
    for (metric, split), mean in means.items():
        values = cl.AUTHORS[metric][split]
        assert abs(sum(values) / len(values) - mean) < 0.06


def test_committed_report_is_current():
    committed = (ROOT / "results" / "comparison" / "patchcore_literature.md").read_text()
    assert committed == cl.report(_reports())
