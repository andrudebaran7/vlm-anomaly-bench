# Dataset access

## MVTec AD 2 — primary benchmark

1. Download from the official MVTec research page (registration + CC BY-NC-SA 4.0 research license).
2. Register for the evaluation server at https://benchmark.mvtec.com/ — the private test split is
   scored only there.
3. Download the official PyTorch dataset class and utils bundle from the same page. It carries the
   authoritative split names, the submission format, and runtime/memory measurement helpers.

Record on download (these are unverified until then, by design):

- Download size: ____
- Dataset version / date: ____
- Split directory names, read from the official dataset class: ____
- Official metric names, read from the server submission docs: ____

## VisA + MVTec AD (classic) — sanity only

Public downloads, used solely to validate our implementations against published numbers
(±1.0 I-AUROC, protocol §2).

## Real-IAD / Real-IAD Variety — out of scope

Correction to the v0.1 assumption: **no application is required.** The dataset is publicly
accessible at https://huggingface.co/datasets/Real-IAD/Real-IAD under research-only terms.

It is out of scope for compute reasons, not access reasons: 622 GB for the full release, ~53 GB for
the 1024px variant, and Real-IAD Variety comprises 198,950 images across 160 categories. None fits
the Colab budget. Revisit only with a different compute platform (spec §Scope).

## Auxiliary-training data (AnomalyCLIP / AdaCLIP)

Record exactly which auxiliary dataset each official checkpoint was trained on, and verify zero
overlap with MVTec AD 2. This table goes in the paper (§3.1) — it is a common credibility hole in
this literature.
