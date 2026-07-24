import json
from pathlib import Path

import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.mllm import PROMPT, QwenMLLM, cells_to_grid, parse_mllm_response


def _client(response):
    return lambda image, prompt: response


def test_parses_a_clean_response():
    text = json.dumps({"anomaly_probability": 0.82, "cells": ["B3", "B4"], "reason": "scratch"})
    score, cells, ok = parse_mllm_response(text)
    assert ok is True
    assert score == pytest.approx(0.82)
    assert cells == ["B3", "B4"]


def test_finds_json_embedded_in_prose():
    text = 'Sure! Here is my assessment:\n{"anomaly_probability": 0.1, "cells": []}\nHope that helps.'
    score, cells, ok = parse_mllm_response(text)
    assert ok is True and score == pytest.approx(0.1) and cells == []


def test_clamps_an_out_of_range_probability():
    score, _, ok = parse_mllm_response('{"anomaly_probability": 1.7, "cells": []}')
    assert ok is True and score == 1.0


def test_drops_invalid_cell_labels_but_keeps_valid_ones():
    _, cells, ok = parse_mllm_response('{"anomaly_probability": 0.5, "cells": ["B3", "Z9", "h2", "A1"]}')
    assert ok is True and cells == ["B3", "A1"]


def test_unparseable_response_is_flagged_not_guessed():
    score, cells, ok = parse_mllm_response("the image looks fine to me, no JSON here")
    assert ok is False
    assert score == 0.5   # no-information score for AUROC, not a silent 0
    assert cells == []


def test_missing_probability_field_is_a_parse_failure():
    score, cells, ok = parse_mllm_response('{"cells": ["A1"]}')
    assert ok is False and score == 0.5 and cells == []


def test_cells_to_grid_sets_the_named_cells():
    grid = cells_to_grid(["A1", "G7"])
    assert grid.shape == (7, 7)
    assert grid[0, 0] == 1.0 and grid[6, 6] == 1.0
    assert grid.sum() == 2.0


def test_cells_to_grid_empty_is_all_zero():
    assert cells_to_grid([]).sum() == 0.0


def test_predict_scores_and_localises_a_defect():
    text = json.dumps({"anomaly_probability": 0.9, "cells": ["D4"]})
    m = QwenMLLM(model_client=_client(text))
    m.prepare(device="cpu")
    image = np.zeros((70, 70, 3), dtype=np.uint8)
    pred = m.predict(image, "vial")
    assert_valid_prediction(pred, image)
    assert pred.image_score == pytest.approx(0.9)
    assert pred.anomaly_map[35, 35] == 1.0     # centre falls in the D4 block
    assert pred.anomaly_map[0, 0] == 0.0
    assert pred.extras["parse_ok"] is True


def test_predict_on_an_unparseable_response_flags_it_and_stays_valid():
    m = QwenMLLM(model_client=_client("no json"))
    m.prepare(device="cpu")
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    pred = m.predict(image, "vial")
    assert_valid_prediction(pred, image)
    assert pred.image_score == 0.5
    assert (pred.anomaly_map == 0.0).all()
    assert pred.extras["parse_ok"] is False


def test_predict_puts_the_category_in_the_prompt():
    seen = {}
    def client(image, prompt):
        seen["prompt"] = prompt
        return '{"anomaly_probability": 0.0, "cells": []}'
    m = QwenMLLM(model_client=client)
    m.prepare(device="cpu")
    m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "fruit_jelly")
    assert "fruit_jelly" in seen["prompt"]


def test_predict_without_a_client_or_prepare_fails_loudly():
    m = QwenMLLM()  # no injected client, prepare() not called
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "vial")


def test_adapter_prompt_stays_in_sync_with_the_published_config():
    """§3 requires the prompt be published in config for auditability; the adapter embeds its
    own runtime copy to avoid a YAML dependency in CI. This fails if the two drift apart."""
    config = (Path(__file__).resolve().parents[1] / "configs/methods/mllm_qwen.yaml").read_text()
    for line in PROMPT.splitlines():
        assert line.strip() in config, f"prompt line not found in config: {line!r}"
