"""Per-(dataset, method, category) result shards, safe against Colab disconnections.

Writes land on a .tmp path and are renamed into place; os.replace is atomic on POSIX, so a
session killed mid-write leaves either the previous file or nothing — never a truncated shard
that is_done() would wrongly trust and skip.
"""
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


#: Columns a shard owns. They must match the shard's filename, so meta may not set them.
IDENTITY_COLUMNS = ("dataset", "method", "category")


class ResultStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, dataset: str, method: str, category: str) -> Path:
        return self.root / f"{dataset}__{method}__{category}.parquet"

    def is_done(self, dataset: str, method: str, category: str) -> bool:
        return self.path_for(dataset, method, category).is_file()

    def write(
        self,
        dataset: str,
        method: str,
        category: str,
        rows: Sequence[Mapping[str, Any]],
        meta: Mapping[str, Any],
    ) -> Path:
        if not rows:
            raise ValueError(f"refusing to write an empty shard for {dataset}/{method}/{category}")

        # meta used to be applied last, so a meta key named "category" (or
        # dataset/method) silently relabelled every row: the filename still said
        # `can` while the column said something else, and load_all() would group the
        # results under the wrong category with nothing to flag it.
        shadowed = [k for k in IDENTITY_COLUMNS if k in meta]
        if shadowed:
            raise ValueError(
                f"meta keys {shadowed} would overwrite the shard's identity columns "
                f"for {dataset}/{method}/{category}; rename them"
            )

        df = pd.DataFrame(list(rows))
        df["dataset"] = dataset
        df["method"] = method
        df["category"] = category
        for key, value in meta.items():
            df[key] = value

        final = self.path_for(dataset, method, category)
        tmp = final.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(final)
        return final

    def load_all(self) -> pd.DataFrame:
        shards = sorted(self.root.glob("*.parquet"))
        if not shards:
            return pd.DataFrame()
        return pd.concat((pd.read_parquet(p) for p in shards), ignore_index=True)
