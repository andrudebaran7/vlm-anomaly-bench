"""scripts/bootstrap_iauroc.py: the paired, scene-clustered bootstrap (protocol v0.2.26)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score


def _shards(rng, shift=0.0, seeds=(0, 1, 2), conditions=("regular", "shift_1"), n_good=6, n_bad=5):
    rows = []
    for seed in seeds:
        for folder, label, n in (("good", 0, n_good), ("bad", 1, n_bad)):
            for i in range(n):
                for c in conditions:
                    rows.append({"image_path": f"/x/can/test_public/{folder}/{i:03d}_{c}.png",
                                 "split": "test_public", "label": label, "seed": seed,
                                 "meta_lighting": c,
                                 "image_score": float(rng.normal(label + shift, 1.0))})
    rows.append({"image_path": "/x/can/test_private/000_regular.png", "split": "test_private",
                 "label": -1, "seed": 0, "meta_lighting": "regular", "image_score": 9.0})
    return pd.DataFrame(rows)


def test_auroc_equals_sklearn_including_ties():
    from bootstrap_iauroc import auroc

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200)
    scores = np.round(rng.normal(labels, 1.0), 1)          # rounding makes ties
    assert auroc(labels, scores) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)


def test_image_table_keeps_only_labelled_public_images_and_parses_the_scene():
    from bootstrap_iauroc import image_table

    t = image_table(_shards(np.random.default_rng(0)), "antialiased")
    assert set(t["folder"]) == {"good", "bad"} and (t["label"] >= 0).all()
    assert set(t["index"]) == {f"{i:03d}" for i in range(6)}
    assert len(t) == 3 * (6 + 5) * 2


def test_the_point_value_is_the_seed_mean_per_condition_then_the_condition_mean():
    from bootstrap_iauroc import Paired, image_table
    from vlmab.eval.aggregate import image_metrics

    shards = _shards(np.random.default_rng(1))
    p = Paired([image_table(shards, "antialiased")])
    value, conds = p.stat(np.arange(len(p)), "antialiased")
    public = shards[shards["split"] == "test_public"]
    for c, v in conds.items():
        per_seed = [image_metrics(g)["i_auroc"]
                    for _, g in public[public["meta_lighting"] == c].groupby("seed")]
        assert v == pytest.approx(np.mean(per_seed), abs=1e-12)
    assert value == pytest.approx(np.mean(list(conds.values())), abs=1e-12)


def test_a_cluster_is_one_scene_index_across_every_condition():
    from bootstrap_iauroc import Paired, image_table

    p = Paired([image_table(_shards(np.random.default_rng(0)), "antialiased")])
    strata = p.strata()
    assert sorted(len(s) for s in strata) == [5, 6]                     # bad, good clusters
    assert all(len(members) == 2 for s in strata for members in s)      # two conditions each


def test_identical_configurations_give_a_change_of_exactly_zero_in_every_replicate():
    """Pairing: both configurations are scored on the same draw, so equal scores cannot differ."""
    from bootstrap_iauroc import Paired, bootstrap, image_table

    shards = _shards(np.random.default_rng(2))
    p = Paired([image_table(shards, "aliased"), image_table(shards, "antialiased")])
    values, _, _ = bootstrap(p, ["aliased", "antialiased"], 200, np.random.default_rng(0))
    assert np.array_equal(values["aliased"], values["antialiased"])
    assert values["antialiased"].std() > 0                              # the draws do vary


def test_the_bootstrap_is_reproducible_from_its_seed():
    from bootstrap_iauroc import Paired, bootstrap, image_table

    p = Paired([image_table(_shards(np.random.default_rng(3)), "antialiased")])
    a, _, _ = bootstrap(p, ["antialiased"], 100, np.random.default_rng(20261102))
    b, _, _ = bootstrap(p, ["antialiased"], 100, np.random.default_rng(20261102))
    assert np.array_equal(a["antialiased"], b["antialiased"])


def test_an_image_scored_under_one_configuration_only_is_refused():
    from bootstrap_iauroc import Paired, image_table

    shards = _shards(np.random.default_rng(0))
    fewer = shards[~shards["image_path"].str.endswith("004_regular.png")]
    with pytest.raises(ValueError, match="one configuration or seed"):
        Paired([image_table(shards, "aliased"), image_table(fewer, "antialiased")])


def test_a_draw_that_leaves_a_condition_one_class_is_redrawn_and_counted():
    from bootstrap_iauroc import Paired, bootstrap, image_table

    p = Paired([image_table(_shards(np.random.default_rng(0), n_bad=1), "antialiased")])
    values, _, redraws = bootstrap(p, ["antialiased"], 50, np.random.default_rng(0))
    assert redraws == 0                    # one bad cluster is always drawn: stratified by folder
    assert np.isfinite(values["antialiased"]).all()


def test_the_report_check_names_the_condition_that_does_not_reproduce():
    from bootstrap_iauroc import Paired, check_against_report, image_table

    p = Paired([image_table(_shards(np.random.default_rng(4)), "antialiased")])
    _, conds = p.stat(np.arange(len(p)), "antialiased")
    good = (f"| regular | {conds['regular']:.4f} ± 0.0100 | 0.2 ± 0.0 | 0.1 ± 0.0 |\n"
            f"| shift_1 | {conds['shift_1']:.4f} ± 0.0100 | 0.2 ± 0.0 | 0.1 ± 0.0 |\n")
    assert check_against_report(p, "antialiased", good) == []
    bad = good.replace(f"{conds['shift_1']:.4f}", "0.0001")
    assert [m.split(":")[0] for m in check_against_report(p, "antialiased", bad)] == ["shift_1"]


def test_a_shard_directory_with_no_shards_counts_as_absent(tmp_path):
    """An empty `shards/` must take the 'anchor interval only' path, not crash the whole run."""
    from bootstrap_iauroc import load

    (tmp_path / "fruit_jelly" / "shards").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no .*fruit_jelly/shards"):
        load(tmp_path, "fruit_jelly", "aliased")
