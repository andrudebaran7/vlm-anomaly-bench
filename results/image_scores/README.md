# Image-level scores, MVTec AD 2 `test_public`

The PatchCore anchor's score for every test image, per seed, for both configurations: `antialiased` (the
corrected one, v0.2.21) and `aliased` (the one first run). Written by `scripts/export_image_scores.py` from the
result shards; each configuration first had to reproduce its committed report. Fruit Jelly has no `aliased` file,
because those shards no longer exist.

| Column | Meaning |
|---|---|
| `folder` | `good` or `bad` |
| `index` | scene index, with its leading zeros |
| `condition` | lighting condition |
| `label` | 0 normal, 1 anomalous |
| `seed` | 0, 1 or 2 |
| `score` | the image score, at full float precision |

The intervals can be recomputed from these files alone:

    python scripts/bootstrap_iauroc.py --antialiased results/image_scores --aliased results/image_scores
    python scripts/scene_sensitivity.py --antialiased results/image_scores --aliased results/image_scores

## Licence

**These files are not under the repository's MIT licence.** They are derived from MVTec AD 2,
© 2025 MVTec Software GmbH, licensed under CC BY-NC-SA 4.0
(<https://creativecommons.org/licenses/by-nc-sa/4.0/>), and are released under that same licence: attribution,
non-commercial use, share alike. They contain no images. Changes made: each image was reduced to one anomaly score.
Non-commercial use only; the dataset itself comes from MVTec.
