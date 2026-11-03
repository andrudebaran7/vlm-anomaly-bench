"""scripts/scene_sensitivity.py: how far a paired change rests on single scenes (protocol v0.2.27)."""
import numpy as np
import pandas as pd
import pytest


def _shards(rng, shift=0.0, n_good=4, n_bad=5, conditions=("regular", "shift_1"), seeds=(0, 1, 2)):
    rows = []
    for seed in seeds:
        for folder, label, n in (("good", 0, n_good), ("bad", 1, n_bad)):
            for i in range(n):
                for c in conditions:
                    rows.append({"image_path": f"/x/test_public/{folder}/{i:03d}_{c}.png",
                                 "split": "test_public", "label": label, "seed": seed,
                                 "meta_lighting": c,
                                 "image_score": float(rng.normal(label + shift, 1.0))})
    return pd.DataFrame(rows)


def _paired(a, b=None):
    from bootstrap_iauroc import Paired, image_table

    tables = [image_table(a, "antialiased")]
    if b is not None:
        tables.append(image_table(b, "aliased"))
    return Paired(tables)


def test_each_scene_is_left_out_once_and_the_value_is_the_statistic_without_it():
    from scene_sensitivity import leave_one_out

    p = _paired(_shards(np.random.default_rng(0)), _shards(np.random.default_rng(1)))
    rows = leave_one_out(p)
    assert len(rows) == 4 + 5
    for r in rows:
        keep = ~((p.folder == r["folder"]) & (p.index == r["index"]))
        kept = np.flatnonzero(keep)
        assert len(kept) == len(p) - 2                       # one image per condition removed
        assert r["antialiased"] == pytest.approx(p.stat(kept, "antialiased")[0], abs=1e-12)
        assert r["aliased"] == pytest.approx(p.stat(kept, "aliased")[0], abs=1e-12)
        assert r["change"] == pytest.approx(r["antialiased"] - r["aliased"], abs=1e-12)


def test_identical_configurations_change_by_exactly_zero_whichever_scene_is_left_out():
    from scene_sensitivity import leave_one_out

    shards = _shards(np.random.default_rng(2))
    assert all(r["change"] == 0.0 for r in leave_one_out(_paired(shards, shards)))


@pytest.mark.parametrize("folder", ["good", "bad"])
def test_the_scene_shares_of_either_class_average_to_the_category_value(folder):
    """A share is the fraction of opposite-class images a scene ranks correctly, so each class's
    shares average to the I-AUROC: they locate the value in scenes and cannot add to it."""
    from scene_sensitivity import scene_shares

    p = _paired(_shards(np.random.default_rng(3)))
    shares = [s["share"] for s in scene_shares(p, "antialiased") if s["folder"] == folder]
    assert np.mean(shares) == pytest.approx(p.stat(np.arange(len(p)), "antialiased")[0], abs=1e-12)


def test_a_share_counts_a_tie_as_half():
    from scene_sensitivity import scene_shares

    shards = _shards(np.random.default_rng(4), n_good=1, n_bad=1, conditions=("regular",), seeds=(0,))
    shards["image_score"] = 1.0
    assert [s["share"] for s in scene_shares(_paired(shards), "antialiased")] == [0.5, 0.5]


def test_a_scenes_mean_score_is_over_its_conditions_and_seeds():
    from scene_sensitivity import scene_shares

    shards = _shards(np.random.default_rng(5))
    out = {(s["folder"], s["index"]): s["score"] for s in scene_shares(_paired(shards), "antialiased")}
    assert out["good", "002"] == pytest.approx(
        shards[shards["image_path"].str.contains("good/002_")]["image_score"].mean(), abs=1e-12)


@pytest.mark.parametrize("full, changes, verdict", [
    (0.30, [0.20, 0.31, 0.15], False),      # exactly half is not below half
    (0.30, [0.20, 0.149], True),            # below half
    (0.30, [0.20, -0.01], True),            # sign flips
    (-0.30, [-0.20, -0.16], False),         # the rule is about size, in either direction
    (-0.30, [-0.20, -0.10], True),
    (0.30, [0.20, 0.0], True),              # zero is below half
])
def test_the_reading_rule_is_the_one_the_protocol_fixes(full, changes, verdict):
    from scene_sensitivity import scene_dependent

    assert scene_dependent(full, changes) is verdict


def test_the_mechanism_reading_needs_normal_scores_to_fall_and_by_more_than_defect_scores():
    from scene_sensitivity import normal_scores_inflated

    assert normal_scores_inflated(delta_good=-2.0, delta_bad=-1.0) is True
    assert normal_scores_inflated(delta_good=-2.0, delta_bad=1.0) is True
    assert normal_scores_inflated(delta_good=-1.0, delta_bad=-2.0) is False
    assert normal_scores_inflated(delta_good=1.0, delta_bad=2.0) is False


def test_class_means_are_per_folder_over_every_image_condition_and_seed():
    from scene_sensitivity import class_means

    shards = _shards(np.random.default_rng(6))
    means = class_means(_paired(shards), "antialiased")
    public = shards.assign(folder=shards["image_path"].str.extract(r"/(good|bad)/")[0])
    assert means["good"] == pytest.approx(public[public["folder"] == "good"]["image_score"].mean())
    assert means["bad"] == pytest.approx(public[public["folder"] == "bad"]["image_score"].mean())
