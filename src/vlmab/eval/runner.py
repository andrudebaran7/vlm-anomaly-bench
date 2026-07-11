"""Evaluation runner: (dataset x method) -> per-sample parquet in results/.

TODO(M2): iterate samples, call method.predict, persist scores/maps refs, then compute
metrics per category and aggregate. Every output row carries config hash + seed + commit.
"""
