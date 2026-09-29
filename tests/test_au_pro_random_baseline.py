"""What AU-PRO looks like when the anomaly map carries no information.

A metric without a reference point is unreadable. I-AUROC has an obvious one — 0.5 — and AU-PRO
does not, which left every AU-PRO number in this study without an answer to "is that good?". On
2026-09-29 a wallplugs lighting condition came in at `au_pro_005 = 0.0292` and the question could
not be answered from the number alone.

It is derivable: for an uninformative map, per-region overlap tracks the false-positive rate, so
PRO(fpr) ~ fpr and the area normalised over [0, L] is L/2 — 0.15 at the 30% limit, 0.025 at 5%.
This pins the derivation to the actual implementation, so a future change to thresholding,
normalisation or connectivity cannot silently move the reference point that published numbers are
read against.
"""
import numpy as np
import pytest

from vlmab.metrics.pixel_level import au_pro

N_IMAGES = 15
SIDE = 256
N_REGIONS = 3
RADIUS = 12


def _uninformative_batch(rng):
    """Masks with several equal-sized regions, and anomaly maps that are pure noise."""
    masks, amaps = [], []
    yy, xx = np.ogrid[:SIDE, :SIDE]
    for _ in range(N_IMAGES):
        mask = np.zeros((SIDE, SIDE), dtype=np.uint8)
        for _ in range(N_REGIONS):
            cy, cx = rng.integers(RADIUS, SIDE - RADIUS, size=2)
            mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= RADIUS**2] = 1
        masks.append(mask)
        amaps.append(rng.random((SIDE, SIDE), dtype=np.float32))
    return masks, amaps


@pytest.mark.parametrize("fpr_limit", [0.3, 0.05])
def test_a_random_map_scores_half_the_fpr_limit(fpr_limit):
    """The reference point published AU-PRO numbers are read against."""
    scores = []
    for seed in range(5):
        masks, amaps = _uninformative_batch(np.random.default_rng(seed))
        scores.append(au_pro(masks, amaps, fpr_limit=fpr_limit))
    mean = float(np.mean(scores))
    # 5% of the expected value: wide enough for five trials of fifteen images, narrow enough that
    # a change in normalisation or threshold placement fails here rather than in a paper table.
    assert mean == pytest.approx(fpr_limit / 2, rel=0.05), (
        f"AU-PRO at fpr_limit={fpr_limit} scored {mean:.4f} on uninformative maps, not the "
        f"{fpr_limit / 2:.4f} the metric's definition implies. Either the normalisation changed "
        "or the threshold sweep did; every AU-PRO number in results/ is read against this."
    )
