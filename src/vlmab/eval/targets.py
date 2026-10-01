"""Which block of a reproduction config holds a dataset's target.

Hardcoding the name per method would need editing for every method added and would pick the
wrong block silently if a name changed. Each block already states its `dataset`, so the answer
is in the file. Blocks without a `dataset` key -- `not_the_target`, the counts at the top level --
are skipped by construction, which matters: WinCLIP's `not_the_target` holds Table 7's 78.9,
higher than the real 78.1 and therefore the number a wrong lookup would be tempted by.
"""
from pathlib import Path
from typing import Any

import yaml


def block_for(targets: Path, dataset: str) -> str:
    """The single top-level block in `targets` whose `dataset` is `dataset`.

    Raises ValueError for none or several, naming what was found. Guessing between two blocks
    would mean scoring a gate against a target nobody chose.
    """
    loaded: dict[str, Any] = yaml.safe_load(Path(targets).read_text()) or {}
    blocks = {
        name: body for name, body in loaded.items()
        if isinstance(body, dict) and "dataset" in body
    }
    matching = sorted(name for name, body in blocks.items() if body["dataset"] == dataset)
    if not matching:
        raise ValueError(
            f"{Path(targets).name} has no block for dataset {dataset!r}; "
            f"blocks that declare one: {sorted(blocks)}"
        )
    if len(matching) > 1:
        raise ValueError(
            f"{Path(targets).name} has {len(matching)} blocks for dataset {dataset!r} "
            f"({matching}); a gate scored against a guessed block is scored against a target "
            "nobody chose"
        )
    return matching[0]
