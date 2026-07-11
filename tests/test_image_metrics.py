import numpy as np
from vlmab.metrics.image_level import i_auroc, i_ap, i_f1max


def test_perfect_separation():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert i_auroc(labels, scores) == 1.0
    assert i_ap(labels, scores) == 1.0
    assert i_f1max(labels, scores) == 1.0


def test_random_scores_auroc_half():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 2000)
    scores = rng.random(2000)
    assert abs(i_auroc(labels, scores) - 0.5) < 0.05
