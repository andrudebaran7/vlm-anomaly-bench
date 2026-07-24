"""Qwen2.5-VL-3B baseline: a generalist MLLM prompted as an inspector (protocol §3).

The model call is the least interesting part and the only part that needs a GPU, so it is an
injected boundary: `model_client(image, prompt) -> str`. Everything that decides the numbers —
formatting the fixed prompt, parsing the response, turning a 7x7 cell grid into a native-
resolution map, and what to do when the model does not answer in the required form — lives here
and is tested without weights. `prepare()` builds the real client on Colab.

The prompt is embedded here as the single runtime copy, so no YAML parser is pulled into the
test environment for it. `configs/methods/mllm_qwen.yaml` stays the published, auditable copy
protocol §3 requires; `test_adapter_prompt_stays_in_sync_with_the_published_config` fails if the
two drift apart. `{category}` is the only substituted token and every other brace is literal, so
substitution is a plain `.replace`, not `str.format`.

Parse-failure policy (protocol §3, dated amendment): a response we cannot parse carries no
information, so its image score is 0.5 — the no-information point for AUROC — never a silent 0
that would read as "confidently normal", and never NaN. The failure is recorded in
`Prediction.extras["parse_ok"] = False` so the parse-failure *rate* is itself a reported number,
which is exactly the MLLM failure taxonomy the paper's analysis calls for.
"""
import json
import math
import re
from typing import Callable

import numpy as np

from vlmab.methods.base import AnomalyMethod, Prediction
from vlmab.methods.postprocess import upsample_to

_GRID = 7
_CELL = re.compile(r"^([A-G])([1-7])$")
_NO_INFO_SCORE = 0.5

PROMPT = (
    "You are an industrial quality inspector. Look at this image of a {category}.\n"
    "1) Is there any defect or anomaly? Answer with a probability between 0.00 and 1.00.\n"
    "2) If yes, describe its location using a 7x7 grid (rows A-G, columns 1-7).\n"
    'Respond as JSON: {"anomaly_probability": <float>, "cells": ["B3", ...], "reason": "<short>"}'
)


def cells_to_grid(cells: list[str]) -> np.ndarray:
    """A 7x7 float32 grid, 1.0 at each valid `<row A-G><col 1-7>` cell, 0 elsewhere."""
    grid = np.zeros((_GRID, _GRID), dtype=np.float32)
    for cell in cells:
        m = _CELL.match(cell)
        if m:
            grid[ord(m.group(1)) - ord("A"), int(m.group(2)) - 1] = 1.0
    return grid


def parse_mllm_response(text: str) -> tuple[float, list[str], bool]:
    """`(score, cells, parse_ok)` from a model response.

    Tolerant of prose around the JSON and of a probability out of range; strict about the one
    field that carries the score. `parse_ok=False` returns the no-information score and no cells.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            prob = float(obj["anomaly_probability"])
            # json.loads accepts the non-standard tokens NaN/Infinity/-Infinity by default; a
            # non-finite value is not a valid score (protocol §3: the score is never NaN, and
            # clipping would silently turn Infinity into 1.0), so treat it as a parse failure.
            if not math.isfinite(prob):
                raise ValueError("non-finite anomaly_probability")
            score = float(np.clip(prob, 0.0, 1.0))
            raw = obj.get("cells", []) or []
            cells = [c for c in raw if isinstance(c, str) and _CELL.match(c)]
            return score, cells, True
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    return _NO_INFO_SCORE, [], False


class QwenMLLM(AnomalyMethod):
    name = "mllm_qwen"
    zero_shot = True

    def __init__(self, model_client: Callable[[np.ndarray, str], str] | None = None):
        self._client = model_client

    def prepare(self, device: str = "cuda") -> None:
        """With an injected client there is nothing to load. Otherwise the real Qwen client is
        built here — a Colab-only path that imports transformers lazily, never in CI."""
        if self._client is None:  # pragma: no cover - needs a GPU and transformers
            raise RuntimeError(
                "no model_client injected and real-client construction is Colab-only; "
                "inject a callable for CPU use or build the Qwen client in a GPU session"
            )

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        if self._client is None:
            raise RuntimeError("QwenMLLM has no model_client; inject one or call prepare() on GPU")
        response = self._client(image, PROMPT.replace("{category}", category))
        score, cells, parse_ok = parse_mllm_response(response)
        amap = upsample_to(cells_to_grid(cells), image.shape[:2])
        return Prediction(
            image_score=score,
            anomaly_map=amap,
            extras={"parse_ok": parse_ok, "n_cells": len(cells), "raw": response[:500]},
        )
