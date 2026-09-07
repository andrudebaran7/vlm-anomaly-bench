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

#: Per-sample columns the runner writes. meta is stamped on afterwards and is one constant
#: for the whole shard, so a meta key with one of these names would replace real per-sample
#: data with that constant. Listed explicitly (rather than relying only on the keys the rows
#: happen to carry) so the guard also covers a column a run legitimately omitted — a shard
#: written without maps has no `map_path`, and a `map_path` in meta must still be refused.
ROW_SCHEMA_COLUMNS = ("image_path", "label", "image_score", "split", "mask_path", "map_path")


def seed_tag(seed: int | None) -> str:
    """The literal that separates one seed's shard and maps from another's.

    Deliberately a readable literal rather than a component of `config_hash`: the seed is a
    property of one execution, not of the configuration, and a directory name that changed
    because a hash moved cannot be audited by eye. `unseeded` is a real state — the run applied
    no seed — and is not the same as seed 0.
    """
    return "unseeded" if seed is None else f"seed{seed}"


class ResultStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, dataset: str, method: str, category: str, seed: int | None) -> Path:
        return self.root / f"{dataset}__{method}__{category}__{seed_tag(seed)}.parquet"

    def is_done(self, dataset: str, method: str, category: str, seed: int | None) -> bool:
        return self.path_for(dataset, method, category, seed).is_file()

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

        # meta is applied last, so any meta key that names an existing column silently
        # replaces that column with one constant. For the identity columns that means the
        # filename still says `can` while the column says something else, and load_all()
        # groups the results under the wrong category. For the per-sample columns it is
        # worse: a meta key named `label` or `image_score` overwrites the ground truth or
        # the predictions themselves with a constant, and the shard still aggregates
        # cleanly into published metrics computed from that constant.
        row_columns = {k for row in rows for k in row}
        protected = set(IDENTITY_COLUMNS) | set(ROW_SCHEMA_COLUMNS) | row_columns
        shadowed = sorted(k for k in meta if k in protected)
        if shadowed:
            raise ValueError(
                f"meta keys {shadowed} would overwrite columns the shard already owns "
                f"for {dataset}/{method}/{category} (identity columns "
                f"{list(IDENTITY_COLUMNS)}, per-sample columns "
                f"{sorted(set(ROW_SCHEMA_COLUMNS) | row_columns)}); rename them"
            )

        df = pd.DataFrame(list(rows))
        df["dataset"] = dataset
        df["method"] = method
        df["category"] = category
        for key, value in meta.items():
            df[key] = value

        # The seed comes from the meta this shard is about to record, so the filename and the
        # `seed` column cannot disagree. A separate seed argument would be a second value to
        # keep in step, which is the class of bug this whole change removes.
        final = self.path_for(dataset, method, category, meta.get("seed"))
        tmp = final.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(final)
        return final

    def load_all(self) -> pd.DataFrame:
        shards = sorted(self.root.glob("*.parquet"))
        if not shards:
            return pd.DataFrame()
        return pd.concat((pd.read_parquet(p) for p in shards), ignore_index=True)
