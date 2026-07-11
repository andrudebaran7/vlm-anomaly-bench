# Dataset access — do this FIRST (M1)

Access is the long pole of this project: one dataset requires an application with unknown lead time.
Start both processes before writing any code.

## Real-IAD / Real-IAD Variety  ⚠️ application required

1. Go to the official Real-IAD project page (github.com/Tencent → Real-IAD, or the project site
   linked from the CVPR 2024 paper) and locate the dataset application form.
2. Apply with academic/research purpose; mention the published survey (Zenodo DOI) as context.
3. Record here: date sent, contact, response. **If no response in 2 weeks, follow up and consider
   the paper's scope-B fallback: MVTec AD 2 + VisA-challenge splits only.**

Status: ☐ not sent · sent on: ____ · approved on: ____

## MVTec AD 2

1. Download from the official MVTec research page (registration + research license).
2. Note the license terms: no redistribution; cite their paper; check whether test GT is public or
   evaluation-server based — this changes our tooling (see protocol TODO).
3. Verify checksums and record dataset version here.

Status: ☐ not downloaded · version/date: ____

## VisA + MVTec AD (classic)

Public downloads; used for reproduction sanity checks only. Standard sources (AWS mirror for VisA,
MVTec page for AD classic).

## Auxiliary-training data (for AnomalyCLIP / AdaCLIP)

These methods train prompts on auxiliary AD data. Record exactly which auxiliary dataset each
official checkpoint used, and verify zero overlap with our test datasets. This table goes in the
paper (§3.1) — it is a common credibility hole in this literature.
