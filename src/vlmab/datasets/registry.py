"""Name -> dataset class, so a CLI can select one without importing all of them.

Kept beside the loaders rather than inside `run_eval.py` because the reproduction gate, the
notebooks and the CLI all need the same mapping, and three copies of it would drift. The names
are the ones stamped into every shard's `dataset` column (`AnomalyDataset.name`), which is what
`scripts/reproduction_gate.py` matches its targets against — so a name here is part of the
result format, not a convenience label.
"""
from typing import Type

from vlmab.datasets.base import AnomalyDataset
from vlmab.datasets.mvtec_ad import MVTecAD
from vlmab.datasets.mvtec_ad2 import MVTecAD2
from vlmab.datasets.visa import VisA

DATASETS: dict[str, Type[AnomalyDataset]] = {
    MVTecAD2.name: MVTecAD2,     # the primary benchmark
    MVTecAD.name: MVTecAD,       # reproduction only (protocol §2)
    VisA.name: VisA,             # reproduction only (protocol §2)
}


def available() -> list[str]:
    return sorted(DATASETS)


def build_dataset(name: str, root) -> AnomalyDataset:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {available()}")
    return DATASETS[name](root)
