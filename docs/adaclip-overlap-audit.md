# AdaCLIP auxiliary-training overlap audit (protocol §3.1)

AdaCLIP is auxiliary-trained: its static prompt component is learned on an anomaly-detection dataset
and fixed in the checkpoint, and a dynamic component is generated per test image. Its numbers are only
valid if the auxiliary training data does not overlap the test set — a common credibility hole in this
literature, which this table closes. It is the companion to docs/anomalyclip-overlap-audit.md, and
follows the same rule deliberately, so the paper's §3.1 table reads consistently across both
auxiliary-trained methods.

## Which checkpoint we use for which test set, and why it is clean

**Updated 2026-09-19**, when the paper and the official repo were read directly (paper repo,
`docs/verified-literature-facts.md`, seventh pass). Two cells changed and one row is new; the
conclusion did not change.

| Our test set | Checkpoint used | Auxiliary data (complete list, **test splits**) | Overlap with the test set? |
|---|---|---|---|
| MVTec AD 2 (primary) | `VisA & ColonDB` | VisA **+ ColonDB** | None — VisA is a different dataset and provider; MVTec AD 2, though MVTec-family, is never touched by a VisA-trained model. ColonDB is colon-polyp endoscopy and cannot overlap an industrial dataset in any sense. |
| VisA (reproduction §2) | `MVTec AD & ClinicDB` | MVTec AD (classic) **+ ClinicDB** | None — MVTec AD classic and VisA are different datasets. ClinicDB is colon-polyp endoscopy. |
| **either** | `All Datasets` | 14 datasets including **both** mvtec and visa | **Total. This checkpoint must never be loaded here** — see below. |

Two things in that table are new and neither was visible from the abstract the earlier version was
written against:

1. **Each checkpoint carries a medical auxiliary dataset as well as an industrial one**
   (arXiv:2407.15795v1 §5.1). A row naming only the industrial half would be describing a
   checkpoint that does not exist.
2. **The auxiliary data is the datasets' TEST split**, with its anomaly labels and masks
   (Appendix §1: "We solely utilize the test data from these datasets"), corroborated by that
   appendix's own image counts. Same practice as AnomalyCLIP, and it has to be, because the
   prompts are learned against labelled anomalies. The conclusion is unchanged — a test split of
   VisA overlaps MVTec AD 2 exactly as little as its train split does — but an auxiliary-training
   overlap audit that does not say which split was trained on is not an audit.

## The checkpoint that must never be loaded

The official repo publishes a third weight, **"All Datasets Mentioned Above"**, whose training
list is `br35h brain_mri btad clinicdb colondb dagm dtd headct isic mpdd mvtec sdd tn3k visa`
(README, "Train"). It is the demo weight behind the repo's HuggingFace Space, and it contains
**both** MVTec AD and VisA. Loading it for either gate, or for MVTec AD 2, would put the test
set's own family into the auxiliary data and silently void the zero-shot claim — and **nothing in
a filename would reveal it**. `configs/reproduction/adaclip.yaml` records it as
`forbidden_checkpoint: all_datasets`, and the sha256 of whatever is actually loaded goes in the
record below.

## A second channel from the test set into the released weights: model selection

**Found 2026-09-19 by reading `train.py`, not the prose, and recorded because this audit had no
slot for it.** The repo's training loop builds its validation loader from `--testing_data` and
saves `_best.pth` whenever `metric_dict['Average']['f1_px']` improves — pixel-level max-F1 on
that set. Its own comment: *"Typically, we use MVTec or VisA as the validation set. The best
model from this validation process is then used for zero-shot anomaly detection on novel
categories."* The README adds that training is FP16 and unstable enough that the recommended
practice is to train repeatedly and keep the best on that validation set.

| Checkpoint | Auxiliary **training** data | Selected by its score on |
|---|---|---|
| `VisA & ColonDB` | VisA + ColonDB | **MVTec AD (classic)** |
| `MVTec AD & ClinicDB` | MVTec AD (classic) + ClinicDB | **VisA** |

So no category of either test set is in the auxiliary *training* data — which is what this audit
asserts, and it remains true — but the released weights were *chosen* using the dataset they are
then evaluated on. For our primary evaluation this means the checkpoint we run on MVTec AD 2 was
selected on MVTec AD classic, i.e. the MVTec family, which is the very proximity the section
below says we picked this checkpoint to avoid.

It does not change `domain_proximity_caveat`, which this document defines over auxiliary
**training** data, and it is not something we can fix by choosing differently: both published
checkpoints were selected this way, and the alternative is training AdaCLIP ourselves, which §3
does not ask for and which would replace an attributable checkpoint with an unattributable one.
**It is a disclosure, and it belongs in the paper's §3.1 table as its own line rather than folded
into the overlap verdict.** Recorded here on the day it was found, before any AdaCLIP number
exists.

## The subtlety we deliberately avoid

MVTec AD 2 is a *different dataset* from MVTec AD classic (new images, mostly new object categories),
so even the MVTec-AD-trained checkpoint would not overlap it at the image level. But MVTec AD 2 shares
MVTec AD classic's provider and industrial-inspection domain, which is a softer, domain-proximity form
of leakage. We use the **VisA-trained** checkpoint for MVTec AD 2 precisely to avoid it. This is the
conservative choice; it removes the domain-proximity question rather than arguing it away.

## Contingency: what if only one checkpoint is published? — RESOLVED 2026-09-19

**The contingency fired, in its third branch, and the answer is favourable.** The repo publishes
weights for `MVTec AD & ClinicDB`, for `VisA & ColonDB` and for `All Datasets Mentioned Above`
(README "Weight Preparation", read directly 2026-09-19 at commit
`b762ac40c3f33c77e7e513e48cb436f059d456da`). So both of the checkpoints this audit assumed do
exist — but neither is trained on *exactly* VisA or *exactly* MVTec AD classic, which is the
third branch of the rule below: a mixture including a dataset not enumerated here.

Applying that branch as written, and not one word further:

- The complete auxiliary training-set list for each checkpoint is now in the table above, in
  place of its nearest label.
- `domain_proximity_caveat: false`, for both. Neither ColonDB nor ClinicDB shares a provider or
  an industrial-inspection domain with anything we test on, and the industrial component of each
  is the pairing the clean rows above already declare clean.

**One ambiguity in the rule, recorded rather than resolved silently.** Read at its most literal,
"shares a provider or industrial-inspection domain with the test set in use" would be satisfied
by VisA itself, since VisA *is* industrial inspection — which would set the caveat true for the
very pairing the table's first row declares clean. A rule cannot both declare that row clean and
require a caveat for it, so the operative reading is the one the rest of this document plainly
takes: provider-and-family proximity, the MVTec-classic-to-MVTec-AD-2 kind. Flagged here because
the resolution is a judgement about a pre-registered sentence, and those get written down rather
than made in a GPU session.

The original text of the contingency follows, unaltered.

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

Filled in 2026-09-19 where the repo's own files answer it; the rest still needs the GPU session,
because a sha256 is a property of the file that was actually downloaded and nothing else.

- Repo commit read for this audit: **`b762ac40c3f33c77e7e513e48cb436f059d456da`** (2025-07-07).
  The commit the bench actually clones is pinned in `configs/methods/adaclip.yaml` at the Colab
  run and may differ; if it does, re-read the README's weight table before trusting this one.
- Checkpoints published by the repo (every one found): **`MVTec AD & ClinicDB`**,
  **`VisA & ColonDB`**, **`All Datasets Mentioned Above`** — the third being forbidden here.
- `VisA & ColonDB` checkpoint file + sha256: ____ (used for MVTec AD 2 and for the MVTec AD gate)
- `MVTec AD & ClinicDB` checkpoint file + sha256: ____ (used for the VisA gate)
- Contingency triggered? **other-auxiliary-data** — both checkpoints exist, and both carry a
  medical dataset alongside the industrial one.
- If triggered, the caveat text used in the paper's tables: **no domain-proximity caveat** (see
  the resolution above). The two disclosures AdaCLIP does owe every table it appears in are the
  model-selection channel and the maintainers' own statement that the released weights do not
  reproduce the published table — `released_weights_caveat` in
  `configs/reproduction/adaclip.yaml`.
- Checkpoint the run actually loaded, per gate, recorded from the run itself: ____

## Still owed by the backend, when it is built

A shard names its dataset, method, category, seed and `preprocess`; it has **no field for a
checkpoint** — the gap recorded with AnomalyCLIP's audit, and AdaCLIP has it too, with a third
weight sitting in the same directory that must never be the one loaded. The fix is the same:
declare the checkpoint the way `PatchCoreBackend` declares its seed, and have the adapter refuse
a declared value its backend does not apply.
