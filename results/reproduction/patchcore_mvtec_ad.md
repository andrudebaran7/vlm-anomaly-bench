# Reproduction gate — patchcore_ref on mvtec_ad

Generated 2026-09-16 17:22 UTC by `scripts/reproduction_gate.py`.

## Verdict: **FAIL**

Measured mean I-AUROC **97.94** against published **99.0** (delta -1.06, tolerance ±1.0).

⚠️ **5 of 15 categories are outside ±1.0 on their own:** cable (-2.00), carpet (-1.11), pill (-2.11), tile (+1.10), toothbrush (-8.87). The gate is on the mean, so this does not change the verdict — but it is not an even reproduction, and the cause belongs in the write-up before any frontier number is reported.

Source: arXiv:2106.08265 Table S1, row PatchCore-10

## Per category

| Category | Measured | Published | Delta |
|---|---|---|---|
| bottle | 100.00 | 100.0 | +0.00 |
| cable | 97.40 | 99.4 | -2.00 ⚠️ |
| capsule | 98.44 | 97.8 | +0.64 |
| carpet | 97.59 | 98.7 | -1.11 ⚠️ |
| grid | 97.91 | 97.9 | +0.01 |
| hazelnut | 100.00 | 100.0 | +0.00 |
| leather | 100.00 | 100.0 | +0.00 |
| metal_nut | 99.17 | 100.0 | -0.83 |
| pill | 93.89 | 96.0 | -2.11 ⚠️ |
| screw | 97.32 | 97.0 | +0.32 |
| tile | 100.00 | 98.9 | +1.10 ⚠️ |
| toothbrush | 90.83 | 99.7 | -8.87 ⚠️ |
| transistor | 99.29 | 100.0 | -0.71 |
| wood | 98.77 | 99.0 | -0.23 |
| zipper | 98.53 | 99.5 | -0.97 |

## Provenance

- results_root: `results/reproduction/mvtec_ad/shards`
- targets: `configs/reproduction/patchcore_ref.yaml`
- n_categories: `15`
- commit: `647ca0b1b4a319091f1c691e5875ff6257d1f469`
- seed: `0`
- gpu: `Tesla T4`

---

## Committed verbatim, 2026-09-16 — this is the record, not a draft

Written back into the repo from the Colab session's output. Protocol §2 requires the gate's
result to exist whether it passed or not; a FAIL that lives only in a closed session's scrollback
is the same as no gate at all.

**Not yet a final verdict on the method**, for one reason fixed in the protocol before this ran:
§6 requires three seeds where stochasticity exists, and PatchCore is stochastic (measured
2026-08-26). This is **one seed**. A miss of 0.057 beyond the tolerance, with the seed spread
unmeasured, is a result awaiting two more runs — not a reproduction failure that can be reported.
See `docs/next-steps.md` for the two pre-registered candidate causes and what is owed next.
