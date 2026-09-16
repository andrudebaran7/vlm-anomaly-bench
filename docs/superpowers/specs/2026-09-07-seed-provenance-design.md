# The seed a shard records is the seed that was applied (spec)

**Date:** 2026-09-07
**Status:** approved, pre-implementation
**Produces:** one implementation plan — `docs/superpowers/plans/` — executable on CPU, no GPU session.
**Amends:** nothing in `docs/protocol.md`. §6 already requires three seeds; the protocol does not
specify shard filenames, and no results are committed, so this needs no Changelog entry. §6 gains one
sentence saying where the seed now lives.
**Blocks:** the PatchCore VisA ±1pt gate (phase 3). `docs/next-steps.md` states the ordering: a gate
run once, unseeded, under a `seed: 0` record is not the pre-registered procedure.

## The defect

`scripts/run_eval.py` accepts `--seed`, hands it to `run_meta()`, and writes it into every shard's
provenance **without applying it to anything**. For `intensity_baseline` that was harmless — it is
deterministic. PatchCore is the first stochastic method here: its greedy coreset sampling gave
202.796432 and then 203.890701 for the same image across two fits on byte-identical inputs (probe
stage 11, 2026-08-26). A shard would record `seed: 0` over a memory bank sampled with no seed at all.
That is provenance that reads true and is not, which is worse than provenance that is absent.

Three further facts, found while reading the code for this spec and not previously recorded:

1. **The seed is not part of a run's identity.** `runner.run_id(meta)` returns `meta["config_hash"]`,
   and both entry points build the config as `{"method", "split"}` — no seed. Two seeds therefore
   produce the same `run_id` and the same anomaly-map directory. That is exactly the collision
   `run_id`'s docstring was written to prevent: the second run overwrites the first run's maps while
   the first run's completed shard still points at them.
2. **The shard path does not distinguish seeds either.** `ResultStore.path_for` is
   `<dataset>__<method>__<category>.parquet`. Running a second seed into the same `--results` root
   either overwrites the first or — because `is_done()` is keyed on that path — is skipped entirely,
   leaving a root that reads as three seeds and is one seed repeated three times.
3. **The entry point that runs on GPU is not the CLI.** Notebook cell 24 builds
   `PatchCoreRef(backend=PatchCoreBackend())` and `run_meta({...}, seed=0)` by hand. A fix confined to
   `scripts/run_eval.py` would leave the only path that actually executes still able to lie.

The root cause is one shape, visible in both entry points: the seed that is *applied* (the backend
constructor) and the seed that is *recorded* (`run_meta`) are two independent values that nothing
reconciles.

## The rule

**A method declares the seed it applied. The runner records what the method declares. Nobody else
gets to write that field.**

One value, one writer. A record cannot disagree with what happened, because there is nothing for it to
disagree with.

## 1. The contract

`AnomalyMethod` gains one declarative attribute:

```python
class AnomalyMethod:
    name: str = "base"
    zero_shot: bool = True
    #: The seed this method applied to its own stochasticity. None means none was applied.
    seed: int | None = None
```

It means *"the seed I applied"*, never *"the seed I was asked for"*. The default `None` is the honest
value for a method that seeds nothing.

**A deterministic method declares `None` and refuses a seed.** `IntensityBaseline.__init__(seed=None)`
raises `ValueError` on a non-`None` seed:

> `intensity_baseline is deterministic; a seed would be recorded and never applied`

Recording `0` there would be the same falsehood in a smaller font. Its reproducibility does not come
from a seed and must not be documented as if it did.

**A method whose stochasticity lives in an injected backend derives its seed from that backend.**
`PatchCoreRef.seed` becomes a read-only `property` — overriding the base class attribute, which is
what a property on a subclass does — returning `getattr(self._backend, "seed", None)`, because the
backend is what calls `seed_everything`. `PatchCoreBackend` exposes `seed` publicly (today
`self._seed`). The `getattr` default matters: the existing tests inject fake backends that have no
`seed` attribute, and those must keep declaring `None` rather than raising. If a caller passes
*both* an injected backend and an explicit `seed=` that disagrees with it, that is a `ValueError`: two
callers each believing they set the seed is the defect this spec exists to remove, and it should not
be resolved by a precedence rule nobody will remember.

With no backend injected, `PatchCoreRef` keeps the constructor's `seed` — it has nothing to
reconcile against, and `prepare()` raises `MethodNotRunnable` before anything is scored anyway.

Every adapter's `__init__` accepts `seed: int | None = None`, so `build_method` can pass one
uniformly without knowing which adapters are stochastic.

## 2. The runner stamps it; `run_meta` cannot

`run_meta(cfg)` loses its `seed` parameter. There is no longer any way for a caller to put a seed into
provenance by hand — including notebook cell 24, which is the point.

`run_evaluation` builds the effective meta itself, immediately before writing:

```python
effective = dict(meta)
if "seed" in effective:
    raise ValueError(
        "meta already carries a seed; the seed is stamped from method.seed, which is the "
        "only value that was actually applied"
    )
effective["seed"] = method.seed
```

The guard is not defensive decoration: a caller assembling meta by hand is exactly how the current
defect got in, and `store.write`'s shadowing check does not cover it because `seed` is a meta key
rather than a row column.

Nothing in the runner decides *what* the seed should be. It only transcribes.

## 3. The seed is literal in both disk paths, and not in the hash

- **Shard:** `<dataset>__<method>__<category>__seed<N>.parquet`, and
  `<dataset>__<method>__<category>__unseeded.parquet` when the seed is `None`.
- **Map directory:** the same `__seed<N>` / `__unseeded` suffix appended to the existing
  `<dataset>__<method>__<run_id>__<category>` name.

`ResultStore.path_for` and `is_done` take the seed as a parameter; `run_evaluation` passes
`method.seed` to both, so resume is per `(category, seed)` and a second seed no longer reads as done.

**`config_hash` is deliberately left alone.** The seed is not part of a configuration — it is a
property of one execution of that configuration — and folding it into a hash would make the map
directory change for a reason no reader can see. A literal `seed0` in a directory name is auditable by
eye; a hash that moved is not. The two runs stay distinguishable because the *paths* distinguish them,
which is what `run_id`'s docstring actually requires.

`load_all()` is unchanged: it globs every parquet in the root, so a root holding three seeds loads all
three into one frame with a `seed` column. Section 4 is what makes that safe.

## 4. The guard against pooling seeds

`aggregate()` over a three-seed frame would compute one set of metrics over 3N rows, treating the same
image scored under three seeds as three independent samples. The number would be wrong, plausible, and
silent — the same failure shape as the double normalisation, which also produced a believable number
on a broken path.

A helper `_require_single_seed(df)` raises when the frame carries more than one distinct seed, and is
called from `image_metrics`, `pixel_metrics` and `threshold_metrics`. This is the pattern
`threshold_metrics` already uses to refuse a frame containing more than one category.

It **tolerates the column being absent**: the metric tests build bare frames to exercise the
mathematics, and every frame produced by `run_evaluation` carries the column by construction.

Combining three seeds into mean ± std (protocol §6) is **out of scope**. It is reporting, it belongs
with `scripts/make_tables.py` (today a `TODO(M5)` stub), and building it now means guessing a table
shape before three real shards exist. After this spec, doing it wrong raises instead of publishing.

## 5. The CLI and the notebook

`build_method(name, seed=None)` forwards the seed to the adapter constructor.
`scripts/run_eval.py` calls `build_method(args.method, seed=args.seed)` and no longer passes a seed to
`run_meta`.

**`--seed` changes its default from `0` to `None`.** With the old default, running
`intensity_baseline` — the only method the CLI can currently run to completion — would trip the
deterministic-method `ValueError` on every invocation. `None` also states the truth for the common
case: no seed was applied.

Notebook cell 24 becomes `PatchCoreRef(backend=PatchCoreBackend(seed=0))` with a seedless
`run_meta({...})`. This is a change to a cell's *contents*, not to the notebook's cell structure, so it
costs the executor nothing beyond re-running cell 1.1 — and phase 2 has not been run yet, so no
existing output is invalidated.

## 6. Testing

Written test-first, per the repo's workflow. The behaviours that must be pinned:

- `run_evaluation` stamps `method.seed` into the shard, for both an `int` and `None`.
- `run_evaluation` raises when the caller's meta already carries a `seed`.
- `run_meta` no longer accepts a `seed` argument.
- Two methods differing only in declared seed write two shards, and `is_done` reports the second as
  not done — the resume trap, asserted directly.
- The two runs write to two different map directories, and the first run's maps still exist after the
  second run — the overwrite from finding 1, asserted directly.
- `image_metrics`, `pixel_metrics` and `threshold_metrics` each raise on a two-seed frame, and each
  still work on a frame with no `seed` column.
- `IntensityBaseline(seed=0)` raises; `IntensityBaseline()` declares `seed is None`.
- `PatchCoreRef(backend=FakeBackend(seed=7)).seed == 7`; a conflicting explicit seed raises; a
  backend with no `seed` attribute at all yields `None` rather than an `AttributeError`.
- `build_method("intensity_baseline", seed=None)` builds; every registered adapter accepts `seed=`.

## What this does not do

- It does not run three seeds for you. `--seeds 0,1,2` and a loop are not built; three invocations
  into one root now work correctly, which is what the gate needs.
- It does not compute mean ± std. See §4.
- It does not make an unseeded stochastic run an error. `PatchCoreBackend(seed=None)` stays legal and
  now records `seed: null` truthfully. Whether the VisA gate refuses a null seed is a question for the
  gate, which does not exist yet; adding a `stochastic: bool` flag to the interface to enforce it now
  would be building a mechanism for a caller that has not been written.
