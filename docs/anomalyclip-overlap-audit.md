# AnomalyCLIP auxiliary-training overlap audit (protocol §3.1)

AnomalyCLIP is auxiliary-trained: it learns object-agnostic prompts on an anomaly-detection dataset,
then scores target images zero-shot. Its numbers are only valid if the auxiliary training data does
not overlap the test set — a common credibility hole in this literature, which this table closes.

## The official checkpoints (from https://github.com/zqhang/AnomalyCLIP)

The repo states: "We test all datasets by training once on MVTec AD. For MVTec AD, AnomalyCLIP is
trained on VisA." So there are two relevant checkpoints:

| Checkpoint | Auxiliary training data |
|---|---|
| MVTec-AD-trained (main) | MVTec AD (classic), **test split** |
| VisA-trained | VisA, **test split** |

**The split is not a detail, and it was added on 2026-09-18 after reading the paper.** The paper
fine-tunes on the auxiliary dataset's *test* data, stated twice, in two places: "we fine-tune AnomalyCLIP using the test data on
MVTec AD and evaluate the ZSAD performance on other datasets. As for MVTec AD, we fine-tune
AomalyCLIP on the test data of VisA" (§4.1, repeated in Appendix A.1), and Appendix B's "Since we
just use the test data of Datasets". It has to be the test split, because object-agnostic prompts
are learned against **labelled anomalies**, which only the test split has.

This does not change a single conclusion below — a *test* split of MVTec AD classic overlaps VisA
exactly as little as its train split does — but an audit of auxiliary-training overlap that does
not say which split was trained on is not an audit. A reader who checks this claim will find the
sentence "we fine-tune ... using the test data" in the source, and they should find it here first.

## Which checkpoint we use for which test set, and why it is clean

| Our test set | Checkpoint used | Auxiliary data | Overlap with the test set? |
|---|---|---|---|
| MVTec AD 2 (primary) | VisA-trained | VisA test split | None — VisA is a different dataset and provider; and MVTec AD 2, though MVTec-family, is never touched by a VisA-trained model. |
| VisA (reproduction §2) | MVTec-AD-trained | MVTec AD classic test split | None — MVTec AD classic and VisA are different datasets. |
| MVTec AD classic (reproduction §2) | VisA-trained | VisA test split | None. **This is the same checkpoint as the primary row**, so this gate exercises the exact configuration the MVTec AD 2 evaluation will run — which is a reason to value it beyond the ±1.0 verdict. |

**Each reproduction gate names its checkpoint** in `configs/reproduction/anomalyclip.yaml`,
because the paper's two published numbers come from the two different checkpoints (VisA 82.1 from
the MVTec-AD-trained model, MVTec AD 91.5 from the VisA-trained one). Scoring a run against the
other checkpoint's number would fail a correct implementation.

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
- Which checkpoint each reproduction run actually loaded: ____

**Nothing mechanical checks that last line yet**, and it is the same shape of defect as the seed
that was recorded but never applied (2026-08-26) and the `preprocess` a shard could not name
(2026-09-17). A shard records dataset, method, category, seed and `preprocess`; it has no field
for a checkpoint. When the AnomalyCLIP backend is built, it should declare the checkpoint it
loaded the way `PatchCoreBackend` declares its seed and pre-processing, and the adapter should
refuse a declared value its backend does not apply.
