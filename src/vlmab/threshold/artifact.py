"""Read and write the committed calibration artifact.

The artifact is the pre-registration. Protocol §4 (v0.2.11) requires a threshold to be fixed
on the defect-free `validation` split and committed *before* any private-split submission, and
a file in git with a date is what makes that claim checkable afterwards -- a submission whose
artifact was committed later than the submission is a violation anyone can detect from the log.
"""
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml

from vlmab.threshold.rules import RULES, Calibration

#: The protocol version this artifact format belongs to. `load_artifact` refuses anything
#: else: a threshold calibrated under different rules is not a threshold this protocol can
#: report, and silently accepting it would put an unpre-registered number in a results table.
ARTIFACT_PROTOCOL_VERSION = "0.2.11"

#: How each rule's fitted scalar is named in the file. transductive_quantile fits nothing.
VALUE_KEY = {"global_quantile": "threshold", "per_image_robust_z": "k"}


def write_artifact(
    path: Path,
    *,
    dataset: str,
    method: str,
    alpha: float,
    calibrated_on: Mapping[str, Any],
    run_id: str,
    categories: Mapping[str, Mapping[str, Calibration]],
    designated_for_submission: str,
) -> Path:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1]: got {alpha}")
    if designated_for_submission not in RULES:
        raise ValueError(
            f"designated_for_submission {designated_for_submission!r} is not a rule; "
            f"expected one of {RULES}"
        )

    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    for category, by_rule in categories.items():
        if designated_for_submission not in by_rule:
            raise ValueError(
                f"designated_for_submission {designated_for_submission!r} is missing from "
                f"category {category!r}, so this artifact cannot produce the submission it names"
            )
        block: dict[str, dict[str, Any]] = {}
        for rule, cal in by_rule.items():
            if rule not in RULES:
                raise ValueError(f"unknown rule {rule!r} in category {category!r}")
            key = VALUE_KEY.get(rule)
            block[rule] = (
                {}
                if key is None
                else {key: float(cal.value), "n_pixels": int(cal.n_pixels),
                      "degenerate_mad": int(cal.degenerate_mad)}
            )
        blocks[category] = block

    document = {
        "dataset": dataset,
        "method": method,
        "protocol_version": ARTIFACT_PROTOCOL_VERSION,
        "alpha": float(alpha),
        "calibrated_on": dict(calibrated_on),
        "run_id": run_id,
        "calibrated_at": date.today().isoformat(),
        "categories": blocks,
        "designated_for_submission": designated_for_submission,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


def load_artifact(path: Path) -> dict[str, Any]:
    """Parse and validate. Raises rather than returning a document a caller would have to
    re-check, since every caller is about to put its numbers in a results table."""
    raw = yaml.safe_load(Path(path).read_text())
    version = raw.get("protocol_version")
    if version != ARTIFACT_PROTOCOL_VERSION:
        raise ValueError(
            f"{path} has protocol_version {version!r} but this code implements "
            f"{ARTIFACT_PROTOCOL_VERSION!r}: a threshold calibrated under different rules is "
            "not reportable under these ones"
        )
    designated = raw.get("designated_for_submission")
    if designated not in RULES:
        raise ValueError(
            f"{path} designated_for_submission is {designated!r}; expected one of {RULES}"
        )
    for category, block in raw.get("categories", {}).items():
        for rule in block:
            if rule not in RULES:
                raise ValueError(f"{path} category {category!r} has unknown rule {rule!r}")
        if designated not in block:
            raise ValueError(
                f"{path} category {category!r} is missing its designated_for_submission rule "
                f"{designated!r}"
            )
    return raw
