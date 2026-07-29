# AdaCLIP auxiliary-training overlap audit (protocol §3.1)

AdaCLIP is auxiliary-trained: its static prompt component is learned on an anomaly-detection dataset
and fixed in the checkpoint, and a dynamic component is generated per test image. Its numbers are only
valid if the auxiliary training data does not overlap the test set — a common credibility hole in this
literature, which this table closes. It is the companion to docs/anomalyclip-overlap-audit.md, and
follows the same rule deliberately, so the paper's §3.1 table reads consistently across both
auxiliary-trained methods.

## Which checkpoint we use for which test set, and why it is clean

| Our test set | Checkpoint used | Auxiliary data | Overlap with the test set? |
|---|---|---|---|
| MVTec AD 2 (primary) | VisA-trained | VisA | None — VisA is a different dataset and provider; MVTec AD 2, though MVTec-family, is never touched by a VisA-trained model. |
| VisA (reproduction §2) | MVTec-AD-trained | MVTec AD (classic) | None — MVTec AD classic and VisA are different datasets. |

## The subtlety we deliberately avoid

MVTec AD 2 is a *different dataset* from MVTec AD classic (new images, mostly new object categories),
so even the MVTec-AD-trained checkpoint would not overlap it at the image level. But MVTec AD 2 shares
MVTec AD classic's provider and industrial-inspection domain, which is a softer, domain-proximity form
of leakage. We use the **VisA-trained** checkpoint for MVTec AD 2 precisely to avoid it. This is the
conservative choice; it removes the domain-proximity question rather than arguing it away.

## Contingency: what if only one checkpoint is published?

The table above assumes AdaCLIP publishes both a VisA-trained and an MVTec-AD-trained checkpoint. That
is **unverified at the time of writing** and is checked in Colab phase A.2.

Pre-registered rule, written before any AdaCLIP number exists:

> If only the MVTec-AD-trained checkpoint is published, we use it for MVTec AD 2 anyway, **and** record
> the domain proximity explicitly: `domain_proximity_caveat: true` in `configs/methods/adaclip.yaml`,
> a filled-in row below, and a footnote on every table in the paper where AdaCLIP appears. We do not
> report it as clean, and we do not drop the method silently.
>
> If, conversely, only a VisA-trained checkpoint is published, the §2 VisA reproduction gate cannot be
> run cleanly (VisA-on-VisA). In that case the gate runs against AdaCLIP's published **MVTec AD
> (classic)** numbers with the VisA-trained checkpoint instead, and results/reproduction/adaclip_visa.md
> is renamed to record which dataset the gate actually ran on.
>
> If, instead, the published checkpoint(s) were trained on auxiliary data that is neither exactly
> VisA nor exactly MVTec AD (classic) — e.g. a mixture including other datasets, or a dataset we have
> not enumerated here — no checkpoint choice above applies as written. In that case: record the
> complete auxiliary training-set list for the checkpoint actually used in the table's "Auxiliary
> data" cell (not just its nearest label), and set `domain_proximity_caveat: true` if **any**
> component of that training data shares a provider or industrial-inspection domain with the test
> set in use, else `false`. This is decided from the training-set list alone, the same way the clean
> rows above are — never improvised in the GPU session.

## Recorded on first Colab run

- Pinned repo commit: ____
- Checkpoints published by the repo (list every one found): ____
- VisA-trained checkpoint file + sha256: ____
- MVTec-AD-trained checkpoint file + sha256: ____
- Contingency triggered? (none / MVTec-only / VisA-only / other-auxiliary-data): ____
- If triggered, the caveat text used in the paper's tables: ____
