"""Read and write the committed calibration artifact.

The artifact is the pre-registration. Protocol §4 (v0.2.11) requires a threshold to be fixed
on the defect-free `validation` split and committed *before* any private-split submission, and
a file in git with a date is what makes that claim checkable afterwards -- a submission whose
artifact was committed later than the submission is a violation anyone can detect from the log.
"""
from datetime import date
from math import isfinite
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

    # Validate alpha: same check as write_artifact
    alpha = raw.get("alpha")
    try:
        alpha_float = float(alpha)
    except (TypeError, ValueError):
        raise ValueError(
            f"{path} has alpha {alpha!r}; alpha must be a number in (0, 1]"
        )
    if not 0.0 < alpha_float <= 1.0:
        raise ValueError(
            f"{path} has alpha {alpha_float}; alpha must be in (0, 1]"
        )

    designated = raw.get("designated_for_submission")
    if designated not in RULES:
        raise ValueError(
            f"{path} designated_for_submission is {designated!r}; expected one of {RULES}"
        )

    # Validate categories: must exist, not be empty, and be a mapping
    categories = raw.get("categories")
    if categories is None:
        raise ValueError(
            f"{path} has no categories key; an artifact that calibrated nothing cannot produce "
            "a submission"
        )
    if not isinstance(categories, dict):
        raise ValueError(
            f"{path} categories must be a mapping, got {type(categories).__name__}"
        )
    if not categories:
        raise ValueError(
            f"{path} has an empty categories mapping; an artifact that calibrated nothing cannot "
            "produce a submission"
        )

    for category, block in categories.items():
        for rule in block:
            if rule not in RULES:
                raise ValueError(f"{path} category {category!r} has unknown rule {rule!r}")
        if designated not in block:
            raise ValueError(
                f"{path} category {category!r} is missing its designated_for_submission rule "
                f"{designated!r}"
            )
        # Validate the fitted scalars, same reasoning as alpha above: the artifact is a
        # hand-editable file in git, and an artifact with a corrupted fitted value is a
        # malformed pre-registration. Left unchecked it fails silently downstream instead of
        # here: a missing `k`/`threshold` surfaces as a bare KeyError from aggregate.py, and a
        # NaN one loads clean and produces `nan` thresholds -- since `>= nan` is always False,
        # every rule then reports SegF1 0.0 and FPR 0.0 rather than raising.
        for rule, rule_block in block.items():
            key = VALUE_KEY.get(rule)
            if key is None:
                continue
            if not isinstance(rule_block, dict) or key not in rule_block:
                raise ValueError(
                    f"{path} category {category!r} rule {rule!r} is missing its fitted scalar "
                    f"{key!r}: a threshold rule cannot produce a threshold without it"
                )
            value = rule_block[key]
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{path} category {category!r} rule {rule!r} field {key!r} is {value!r}, "
                    "not a number"
                )
            if not isfinite(value_f):
                raise ValueError(
                    f"{path} category {category!r} rule {rule!r} field {key!r} is {value_f}, "
                    "not finite: a corrupted fitted scalar produces thresholds that are also "
                    "not finite, and since `>= nan` is always False, every rule would then "
                    "silently report SegF1 0.0 and FPR 0.0 instead of raising here"
                )
    return raw
