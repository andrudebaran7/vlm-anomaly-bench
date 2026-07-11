"""MVTec AD 2 loader.

TODO(M1): implement after download — verify official split names (incl. lighting-variation
test sets) against the dataset docs. Do NOT hardcode from memory. Record dataset version in
docs/datasets-access.md.
"""
from .base import AnomalyDataset


class MVTecAD2(AnomalyDataset):
    name = "mvtec_ad2"
    # TODO(M1): __init__(root), categories(), samples(), load_mask()
