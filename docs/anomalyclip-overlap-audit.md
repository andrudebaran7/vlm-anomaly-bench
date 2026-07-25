# AnomalyCLIP auxiliary-training overlap audit (protocol §3.1)

AnomalyCLIP is auxiliary-trained: it learns object-agnostic prompts on an anomaly-detection dataset,
then scores target images zero-shot. Its numbers are only valid if the auxiliary training data does
not overlap the test set — a common credibility hole in this literature, which this table closes.

## The official checkpoints (from https://github.com/zqhang/AnomalyCLIP)

The repo states: "We test all datasets by training once on MVTec AD. For MVTec AD, AnomalyCLIP is
trained on VisA." So there are two relevant checkpoints:

| Checkpoint | Auxiliary training data |
|---|---|
| MVTec-AD-trained (main) | MVTec AD (classic) |
| VisA-trained | VisA |

## Which checkpoint we use for which test set, and why it is clean

| Our test set | Checkpoint used | Auxiliary data | Overlap with the test set? |
|---|---|---|---|
| MVTec AD 2 (primary) | VisA-trained | VisA | None — VisA is a different dataset and provider; and MVTec AD 2, though MVTec-family, is never touched by a VisA-trained model. |
| VisA (reproduction §2) | MVTec-AD-trained | MVTec AD (classic) | None — MVTec AD classic and VisA are different datasets. |

## The subtlety we deliberately avoid

MVTec AD 2 is a *different dataset* from MVTec AD classic (new images, mostly new object categories),
so even the MVTec-AD-trained checkpoint would not overlap it at the image level. But MVTec AD 2 shares
MVTec AD classic's provider and industrial-inspection domain, which is a softer, domain-proximity
form of leakage. We use the **VisA-trained** checkpoint for MVTec AD 2 precisely to avoid it —
consistent with AnomalyCLIP's own rule of training on VisA for any MVTec-family test. This is the
conservative choice; it removes the domain-proximity question rather than arguing it away.

## Recorded on first Colab run

- Pinned repo commit: ____
- VisA-trained checkpoint file + sha256: ____
- MVTec-AD-trained checkpoint file + sha256: ____
