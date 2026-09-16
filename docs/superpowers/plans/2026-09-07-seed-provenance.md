# Seed provenance implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the seed a shard records be the seed that was actually applied, and keep three seeds
from overwriting or silently pooling into each other.

**Architecture:** One value, one writer. A method declares the seed it applied (`AnomalyMethod.seed`);
`run_evaluation` transcribes that declaration into the shard and refuses a seed from anyone else;
`run_meta` loses its `seed` parameter so there is no second way in. The seed then appears literally in
the shard filename and the anomaly-map directory name, so `is_done()` distinguishes seeds and two
runs never share a map directory. Finally, the metric functions refuse a frame that pools seeds.

**Tech Stack:** Python 3.11/3.13, numpy, pandas, pyarrow (parquet), pytest. No GPU, no torch, no
anomalib — every task in this plan runs in CI.

**Spec:** `docs/superpowers/specs/2026-09-07-seed-provenance-design.md`

## Global Constraints

- **Run the test suite with `.venv/bin/python -m pytest`.** There is no `python` on PATH in this
  environment and the system `python3` has no pytest.
- **Never commit to `master`.** This plan is executed on a feature branch; `git merge --no-ff` at the
  end. The branch `plan-seed-provenance` already holds the spec commit — implementation continues on
  a branch off it (or on it, if the executor prefers a single branch).
- **TDD, one behaviour at a time.** Write the failing test, watch it fail for the stated reason, then
  make it pass. A test that passes before the implementation is not evidence — it means the test does
  not discriminate.
- **`seed` means "the seed this method applied", never "the seed someone asked for."** Every decision
  below follows from that sentence; if a step seems to conflict with it, the step is wrong.
- **`None` is a legal, meaningful seed value** — it means nothing was seeded — and must survive
  round-tripping through parquet. It is not an error and must not be coerced to `0`.
- The protocol frozen at v0.2.11 is **not amended** by this work. Do not touch
  `docs/protocol.md`'s Changelog. §6 gains one descriptive sentence in Task 7 and nothing else.

## File Structure

| File | Responsibility after this plan |
|---|---|
| `src/vlmab/methods/base.py` | The `seed` contract: the declarative attribute, and `BackendSeeded` for adapters whose stochasticity lives in an injected backend. |
| `src/vlmab/methods/baseline.py` | Deterministic; declares `None` and refuses a seed. |
| `src/vlmab/methods/patchcore_ref.py`, `winclip.py`, `anomalyclip.py`, `adaclip.py`, `saa.py`, `mllm.py` | Accept `seed=`, declare via `BackendSeeded`. |
| `src/vlmab/methods/patchcore_backend.py` | Exposes the applied seed publicly as `.seed`. |
| `src/vlmab/methods/registry.py` | `build_method(name, seed=None)` forwards the seed to the constructor. |
| `src/vlmab/eval/provenance.py` | `run_meta(cfg)` — no seed parameter, so provenance has no back door. |
| `src/vlmab/eval/runner.py` | Stamps `method.seed`; refuses a caller-supplied seed; passes the seed to the store and into the map directory name. |
| `src/vlmab/eval/store.py` | `seed_tag()`; seed in the shard filename; `path_for`/`is_done` take the seed. |
| `src/vlmab/eval/aggregate.py` | `_require_single_seed` guard on the three metric entry points. |
| `scripts/run_eval.py` | `--seed` defaults to `None` and reaches the method through `build_method`. |
| `scripts/calibrate_threshold.py` | Refuses to calibrate on shards that pool seeds. |
| `tests/test_method_seed.py` | **New.** The seed contract, across the base class and every adapter. |

---

### Task 1: The seed contract on the base class

**Files:**
- Modify: `src/vlmab/methods/base.py`
- Modify: `src/vlmab/methods/baseline.py`
- Test: `tests/test_method_seed.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `AnomalyMethod.seed: int | None = None`; `BackendSeeded` with
  `_init_seed(self, backend, seed: int | None) -> None` and a read-only `seed` property;
  `IntensityBaseline(seed: int | None = None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_method_seed.py`:

```python
"""The seed contract: a method declares the seed it applied, and nothing else may claim one.

`seed` never means "the seed I was asked for". A method that seeds nothing declares None, which
is why None is the default rather than 0 — recording 0 for a method that never seeded anything is
the exact falsehood this contract exists to remove.
"""
import pytest

from vlmab.methods.base import AnomalyMethod, BackendSeeded
from vlmab.methods.baseline import IntensityBaseline


class _Backend:
    """A backend that applies a seed and says so, like PatchCoreBackend."""

    def __init__(self, seed=None):
        self.seed = seed


class _SeedlessBackend:
    """A backend from before seeds existed — e.g. the fake backends the adapter tests inject.
    It must declare None rather than raise AttributeError."""


class _Adapter(BackendSeeded):
    name = "adapter"

    def __init__(self, backend=None, seed=None):
        self._init_seed(backend, seed)


def test_a_method_declares_no_seed_by_default():
    assert AnomalyMethod.seed is None


def test_an_adapter_declares_the_seed_its_backend_applied():
    assert _Adapter(backend=_Backend(seed=7)).seed == 7


def test_an_adapter_with_a_seedless_backend_declares_none():
    assert _Adapter(backend=_SeedlessBackend()).seed is None


def test_an_adapter_keeps_its_own_seed_when_no_backend_is_injected():
    assert _Adapter(seed=3).seed == 3


def test_a_seed_that_disagrees_with_the_backend_is_refused():
    with pytest.raises(ValueError, match="disagree"):
        _Adapter(backend=_Backend(seed=0), seed=1)


def test_a_seed_matching_the_backend_is_accepted():
    assert _Adapter(backend=_Backend(seed=1), seed=1).seed == 1


def test_the_deterministic_baseline_declares_no_seed():
    assert IntensityBaseline().seed is None


def test_the_deterministic_baseline_refuses_a_seed():
    with pytest.raises(ValueError, match="deterministic"):
        IntensityBaseline(seed=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_method_seed.py -v`
Expected: collection error — `ImportError: cannot import name 'BackendSeeded' from 'vlmab.methods.base'`.

- [ ] **Step 3: Add the contract to `base.py`**

Add `seed` to `AnomalyMethod`, directly under `zero_shot`:

```python
    name: str = "base"
    zero_shot: bool = True
    #: The seed this method applied to its own stochasticity, or None if it applied none.
    #
    # NOT "the seed the caller asked for". `run_evaluation` stamps this value into every shard's
    # provenance, so a method that declares a seed it did not apply produces a record that reads
    # true and is not — which is worse than no record. A deterministic method declares None and
    # refuses a seed; a method whose stochasticity lives in a backend declares the backend's.
    seed: int | None = None
```

Then, after the `AnomalyMethod` class, add:

```python
class BackendSeeded(AnomalyMethod):
    """An adapter whose stochasticity lives in an injected backend.

    The backend is what calls into the library that samples, so the backend is what applies the
    seed and therefore what declares it. The adapter only passes the declaration on.

    Call `_init_seed(backend, seed)` from the adapter's `__init__`.
    """

    _declared_seed: int | None = None

    def _init_seed(self, backend, seed: int | None) -> None:
        if backend is None:
            # Nothing has been constructed that could apply anything, and prepare() will raise
            # MethodNotRunnable before this method scores a sample. Keep what the caller asked
            # for so a backend built from it later has something to be checked against.
            self._declared_seed = seed
            return
        # getattr, not attribute access: the adapter tests inject fake backends written before
        # seeds existed, and those declare None rather than exploding.
        backend_seed = getattr(backend, "seed", None)
        if seed is not None and seed != backend_seed:
            raise ValueError(
                f"{type(self).__name__} was given seed={seed!r} but its backend applies "
                f"seed={backend_seed!r}; the two disagree and this adapter cannot make the "
                "backend use the other one. Seed the backend at construction and leave this "
                "argument out."
            )
        self._declared_seed = backend_seed

    @property
    def seed(self) -> int | None:
        return self._declared_seed
```

- [ ] **Step 4: Make `IntensityBaseline` refuse a seed**

In `src/vlmab/methods/baseline.py`, add an `__init__` above `prepare`:

```python
    def __init__(self, seed: int | None = None):
        # Deterministic: the same image always yields the same deviation map, so there is no
        # sampling for a seed to control. Accepting one would put a number into the shard's
        # provenance that nothing applied — the same falsehood the seed contract exists to
        # remove, just in a method where it looks harmless.
        if seed is not None:
            raise ValueError(
                f"intensity_baseline is deterministic; seed={seed!r} would be recorded in "
                "provenance and never applied. Omit it."
            )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_method_seed.py -v`
Expected: 8 passed.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (313 + 8). Nothing else constructs `IntensityBaseline` with an argument.

- [ ] **Step 7: Commit**

```bash
git add src/vlmab/methods/base.py src/vlmab/methods/baseline.py tests/test_method_seed.py
git commit -m "feat: a method declares the seed it applied

seed means 'the seed I applied', never 'the seed I was asked for'. A
deterministic method declares None and refuses one, because recording a
seed nothing applied is the falsehood this contract removes."
```

---

### Task 2: Every adapter declares its seed

**Files:**
- Modify: `src/vlmab/methods/patchcore_ref.py:28`, `winclip.py:28`, `anomalyclip.py:28`,
  `adaclip.py:33`, `saa.py:36`, `mllm.py:139`
- Modify: `src/vlmab/methods/patchcore_backend.py:47` and `:122-126`
- Test: `tests/test_method_seed.py`, `tests/test_patchcore_ref.py`

**Interfaces:**
- Consumes: `BackendSeeded`, `_init_seed` from Task 1.
- Produces: every adapter constructor accepts `seed: int | None = None`;
  `PatchCoreBackend.seed` is a public attribute.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_method_seed.py`:

```python
from vlmab.methods.adaclip import AdaClipRef
from vlmab.methods.anomalyclip import AnomalyClipRef
from vlmab.methods.mllm import QwenMLLM
from vlmab.methods.patchcore_ref import PatchCoreRef
from vlmab.methods.saa import SaaRef
from vlmab.methods.winclip import WinClipRef

#: Every adapter whose stochasticity will live in an injected backend. They are listed
#: explicitly rather than discovered, so an adapter added without the seed contract fails here
#: instead of silently opting out of it.
_BACKEND_ADAPTERS = [AdaClipRef, AnomalyClipRef, PatchCoreRef, QwenMLLM, SaaRef, WinClipRef]


@pytest.mark.parametrize("cls", _BACKEND_ADAPTERS, ids=lambda c: c.__name__)
def test_every_backend_adapter_accepts_a_seed_and_declares_it(cls):
    assert cls(seed=5).seed == 5


@pytest.mark.parametrize("cls", _BACKEND_ADAPTERS, ids=lambda c: c.__name__)
def test_every_backend_adapter_declares_none_by_default(cls):
    assert cls().seed is None
```

Append to `tests/test_patchcore_ref.py`:

```python
class _SeededBackend(_FakeBackend):
    """A backend that applied a seed, like the real PatchCoreBackend(seed=...)."""

    def __init__(self, seed=7):
        super().__init__()
        self.seed = seed


def test_declares_the_seed_its_backend_applied():
    assert PatchCoreRef(backend=_SeededBackend(seed=7)).seed == 7


def test_a_backend_that_applied_no_seed_declares_none():
    assert PatchCoreRef(backend=_FakeBackend()).seed is None


def test_a_seed_disagreeing_with_the_backend_is_refused():
    """Two callers each believing they set the seed is the defect this contract removes; it is
    not resolved by a precedence rule nobody will remember."""
    with pytest.raises(ValueError, match="disagree"):
        PatchCoreRef(backend=_SeededBackend(seed=0), seed=1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_method_seed.py tests/test_patchcore_ref.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'seed'` for each adapter,
and `AttributeError`/`None` mismatches in the PatchCore tests.

- [ ] **Step 3: Convert the five backend adapters**

For each of `patchcore_ref.py`, `winclip.py`, `anomalyclip.py`, `adaclip.py`, `saa.py`: change the
base class from `AnomalyMethod` to `BackendSeeded`, extend the import, and take a seed. The
PatchCore one in full — the other four are the same two edits with their own extra state left
untouched:

```python
from vlmab.methods.base import BackendSeeded, MethodNotRunnable, Prediction


class PatchCoreRef(BackendSeeded):
    name = "patchcore_ref"
    zero_shot = False

    def __init__(self, backend=None, seed: int | None = None):
        self._backend = backend
        self._fitted_category = None
        # The backend is what calls seed_everything, so it is what declares the seed.
        self._init_seed(backend, seed)
```

`AnomalyMethod` may remain in a module's imports only if something else in that module still
uses it; otherwise drop it so an unused import does not linger.

For `mllm.py`, the injected object is `model_client` rather than `_backend`, and it is the thing
that would sample:

```python
class QwenMLLM(BackendSeeded):
    def __init__(self, model_client: Callable[[np.ndarray, str], str] | None = None,
                 seed: int | None = None):
        self._client = model_client
        self._init_seed(model_client, seed)
```

Keep whatever the existing `__init__` body already assigns; only the base class, the new
parameter and the `_init_seed` call are added. Do not rename existing attributes.

- [ ] **Step 4: Expose the seed on `PatchCoreBackend`**

In `src/vlmab/methods/patchcore_backend.py`, rename the private attribute so the adapter can read
it. Line 47 becomes:

```python
        # Public: PatchCoreRef reads this to declare, in the shard's provenance, the seed that
        # was actually applied. A private name would force the adapter to reach through it.
        self.seed = seed
```

and the use inside `fit` (lines 122-126) becomes:

```python
        if self.seed is not None:
            # workers=True also seeds the DataLoader worker processes, which matters because
            # the coreset sampler draws inside them.
            from lightning import seed_everything
            seed_everything(self.seed, workers=True)
```

Verify no other `self._seed` remains: `grep -n "_seed" src/vlmab/methods/patchcore_backend.py`
must return nothing. This module is never constructed in CI (it needs anomalib and a GPU), so it
is verified by reading plus the grep; Task 7's notebook change exercises it on Colab.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_method_seed.py tests/test_patchcore_ref.py -v`
Expected: all pass.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass. The existing adapter tests inject fake backends with no `seed` attribute, which
the `getattr` default covers.

- [ ] **Step 7: Commit**

```bash
git add src/vlmab/methods tests/test_method_seed.py tests/test_patchcore_ref.py
git commit -m "feat: every adapter declares the seed its backend applied

The backend is what calls seed_everything, so the backend is what declares
the seed; the adapter passes the declaration on. A seed that disagrees with
an injected backend raises rather than picking a winner."
```

---

### Task 3: The runner stamps the seed, and nobody else may

**Files:**
- Modify: `src/vlmab/eval/provenance.py:64-88`
- Modify: `src/vlmab/eval/runner.py` (inside `run_evaluation`)
- Modify: `scripts/run_eval.py:46`
- Test: `tests/test_runner.py`, `tests/test_provenance.py`

**Interfaces:**
- Consumes: `AnomalyMethod.seed` from Task 1.
- Produces: `run_meta(cfg: Mapping[str, Any]) -> dict[str, Any]` (no `seed` parameter);
  `run_evaluation` raises `ValueError` when `meta` carries a `seed`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_runner.py`:

```python
class _SeededMethod(AnomalyMethod):
    """Declares a seed the way a real stochastic adapter does."""

    name = "seeded"
    zero_shot = True
    seed = 7

    def prepare(self, device="cuda"):
        pass

    def predict(self, image, category):
        return Prediction(
            image_score=0.5, anomaly_map=np.full((8, 8), 0.5, dtype=np.float32)
        )


def test_runner_stamps_the_seed_the_method_declares(tmp_path, fake_dataset):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, _SeededMethod(), store, {}, categories=["alpha"])
    df = store.load_all()
    assert list(df["seed"]) == [7, 7, 7]


def test_runner_stamps_none_for_a_method_that_seeded_nothing(
    tmp_path, fake_dataset, counting_method
):
    """None is a real value, not a missing one: it says no seed was applied, which is exactly
    what a deterministic run's provenance should say."""
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {}, categories=["alpha"])
    df = store.load_all()
    assert "seed" in df.columns and df["seed"].isna().all()


def test_runner_refuses_a_seed_supplied_by_the_caller(tmp_path, fake_dataset, counting_method):
    """The original defect: run_meta took a seed from the CLI and stamped it while the method
    applied nothing. There must be exactly one writer of this field."""
    store = ResultStore(tmp_path)
    with pytest.raises(ValueError, match="method.seed"):
        run_evaluation(fake_dataset, counting_method, store, {"seed": 0}, categories=["alpha"])
```

In `tests/test_provenance.py`, change the four `run_meta(..., seed=0)` calls (lines 92, 118, 147,
153) to drop the argument, and add:

```python
def test_run_meta_will_not_take_a_seed():
    """Provenance has no back door: the seed comes from the method that applied it, stamped by
    the runner. A caller who can pass one here can make the record disagree with the run."""
    with pytest.raises(TypeError):
        run_meta({"method": "winclip"}, seed=0)
```

`test_run_meta_carries_all_provenance_fields` (lines 91-94) asserts the returned dict contains a
`seed`. It no longer does. The test becomes:

```python
def test_run_meta_carries_all_provenance_fields():
    meta = run_meta({"method": "winclip"})
    assert set(meta) == {"config_hash", "commit", "gpu"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_runner.py tests/test_provenance.py -q`
Expected: FAIL — `run_meta() missing 1 required positional argument: 'seed'` from the provenance
tests, and the three new runner tests failing on a missing `seed` column / no exception raised.

- [ ] **Step 3: Remove the seed from `run_meta`**

In `src/vlmab/eval/provenance.py`, change the signature to `def run_meta(cfg: Mapping[str, Any]) ->
dict[str, Any]:`, drop `"seed": int(seed),` from the returned dict, and add to the docstring:

```
    The seed is deliberately NOT here. It is stamped by `run_evaluation` from `method.seed` —
    the only value that was actually applied. When this function took a seed, `scripts/run_eval.py`
    passed `--seed` straight through and every shard recorded a number nothing had applied.
```

- [ ] **Step 4: Stamp it in the runner**

In `src/vlmab/eval/runner.py`, as the **first statements of `run_evaluation`** — before `wanted` and
`todo` are computed, not after the `if not todo: return []` early exit. A caller handing in a seed is
wrong whether or not there is work left, and Task 4 makes the `todo` computation itself depend on the
seed:

```python
    # The seed is stamped from the method, which is the only object that knows what was applied.
    # A caller reaching this with a seed of its own is the original defect walking back in:
    # store.write()'s shadowing guard does not catch it, because `seed` is a meta key rather
    # than a row column.
    if "seed" in meta:
        raise ValueError(
            f"meta carries seed={meta['seed']!r}, but the seed is stamped from method.seed "
            f"({method.name} declares {method.seed!r}) — the only value that was actually "
            "applied. Remove it from meta."
        )
    meta = {**meta, "seed": method.seed}
```

Rebinding `meta` locally keeps every later use (`run_id(meta)`, `store.write(..., meta)`) on the
stamped copy without threading a second name through the function. The caller's mapping is not
mutated.

- [ ] **Step 5: Stop passing a seed from the CLI**

In `scripts/run_eval.py:46`: `meta = run_meta({"method": args.method, "split": args.split})`.
Leave `--seed` alone for now; Task 5 wires it to the method.

- [ ] **Step 6: Update the existing runner tests**

Every `run_evaluation(..., {"seed": 0}, ...)` in `tests/test_runner.py` now raises. The meta in
those tests is scaffolding, not the thing under test, so replace it with an empty mapping:

```bash
sed -i 's/, {"seed": 0}/, {}/g' tests/test_runner.py
.venv/bin/python -m pytest tests/test_runner.py -q
```

`grep -n '"seed": 0' tests/test_runner.py` must come back empty afterwards. Line 178's assertion
that the shard's columns include `seed` stays and must still pass — the runner now supplies it.

Do **not** apply this sed to `tests/test_store.py`: `ResultStore.write` still takes arbitrary meta,
and those tests exercise exactly that.

- [ ] **Step 7: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/vlmab/eval/provenance.py src/vlmab/eval/runner.py scripts/run_eval.py tests/
git commit -m "fix: the runner stamps the seed the method applied

run_meta no longer takes a seed, so there is no way to put one into
provenance except through the method that applied it. A caller-supplied
seed in meta now raises instead of being recorded."
```

---

### Task 4: Two seeds never share a shard path or a map directory

**Files:**
- Modify: `src/vlmab/eval/store.py`
- Modify: `src/vlmab/eval/runner.py` (the `is_done` call, the map directory name)
- Test: `tests/test_store.py`, `tests/test_runner.py`, `tests/test_calibrate_threshold.py:95`

**Interfaces:**
- Consumes: the stamped `meta["seed"]` from Task 3.
- Produces: `seed_tag(seed: int | None) -> str` in `vlmab.eval.store`;
  `ResultStore.path_for(dataset, method, category, seed)` and
  `ResultStore.is_done(dataset, method, category, seed)` — the seed is a **required** fourth
  positional argument on both.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_store.py`:

```python
from vlmab.eval.store import seed_tag


def test_seed_tag_is_readable_for_both_kinds_of_run():
    assert seed_tag(0) == "seed0"
    assert seed_tag(12) == "seed12"
    assert seed_tag(None) == "unseeded"


def test_two_seeds_write_two_shards(tmp_path):
    store = ResultStore(tmp_path)
    a = store.write("mvtec_ad2", "patchcore_ref", "vial", _rows(), {"seed": 0})
    b = store.write("mvtec_ad2", "patchcore_ref", "vial", _rows(), {"seed": 1})
    assert a != b
    assert a.name == "mvtec_ad2__patchcore_ref__vial__seed0.parquet"
    assert b.name == "mvtec_ad2__patchcore_ref__vial__seed1.parquet"
    assert a.is_file() and b.is_file()


def test_a_shard_is_named_for_the_seed_it_records(tmp_path):
    """write() takes the path's seed from the meta it is about to stamp, so the name and the
    column cannot disagree — there is no second argument to get out of step."""
    store = ResultStore(tmp_path)
    path = store.write("mvtec_ad2", "patchcore_ref", "vial", _rows(), {"seed": 3})
    assert "seed3" in path.name
    assert set(pd.read_parquet(path)["seed"]) == {3}


def test_a_run_with_no_seed_is_named_unseeded(tmp_path):
    store = ResultStore(tmp_path)
    path = store.write("mvtec_ad2", "intensity_baseline", "vial", _rows(), {"seed": None})
    assert path.name == "mvtec_ad2__intensity_baseline__vial__unseeded.parquet"


def test_is_done_is_scoped_per_seed(tmp_path):
    """The resume trap: without this, a second seed's run finds the first seed's shard, calls
    the category done, skips it, and leaves a root that reads as two seeds and is one."""
    store = ResultStore(tmp_path)
    store.write("mvtec_ad2", "patchcore_ref", "vial", _rows(), {"seed": 0})
    assert store.is_done("mvtec_ad2", "patchcore_ref", "vial", 0) is True
    assert store.is_done("mvtec_ad2", "patchcore_ref", "vial", 1) is False
    assert store.is_done("mvtec_ad2", "patchcore_ref", "vial", None) is False
```

Append to `tests/test_runner.py`:

```python
class _OtherSeedMethod(_SeededMethod):
    """Same method, different seed, and a DISTINGUISHABLE map — otherwise "the first run's maps
    survived" passes trivially because both runs wrote identical arrays."""

    seed = 8

    def predict(self, image, category):
        return Prediction(
            image_score=0.5, anomaly_map=np.full((8, 8), 0.99, dtype=np.float32)
        )


def test_a_second_seed_is_not_mistaken_for_a_finished_run(tmp_path, fake_dataset):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, _SeededMethod(), store, {}, categories=["alpha"])
    written = run_evaluation(fake_dataset, _OtherSeedMethod(), store, {}, categories=["alpha"])
    assert [p.name for p in written] == ["fake__seeded__alpha__seed8.parquet"]
    assert sorted(store.load_all()["seed"].unique()) == [7, 8]


def test_two_seeds_do_not_share_a_map_directory(tmp_path, fake_dataset):
    """run_id is config_hash, which carries no seed, so two seeds of one config resolved to one
    map directory: the second run overwrote the first run's maps while the first run's finished
    shard still pointed at them."""
    maps = tmp_path / "maps"
    store = ResultStore(tmp_path / "r")
    run_evaluation(fake_dataset, _SeededMethod(), store, {"config_hash": "cfg"},
                   categories=["alpha"], maps_dir=maps)
    first = {p: np.load(p).copy() for p in store.load_all()["map_path"]}
    assert first and not np.allclose(list(first.values())[0], 0.99)

    run_evaluation(fake_dataset, _OtherSeedMethod(), store, {"config_hash": "cfg"},
                   categories=["alpha"], maps_dir=maps)

    for path, data in first.items():
        assert Path(path).exists(), "the second seed deleted the first seed's maps"
        assert np.array_equal(np.load(path), data), "the second seed overwrote them"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_store.py tests/test_runner.py -q`
Expected: FAIL — `ImportError: cannot import name 'seed_tag'`, and the runner tests failing because
both seeds resolve to one shard path and one map directory.

- [ ] **Step 3: Put the seed in the shard filename**

In `src/vlmab/eval/store.py`, add above the `ResultStore` class:

```python
def seed_tag(seed: int | None) -> str:
    """The literal that separates one seed's shard and maps from another's.

    Deliberately a readable literal rather than a component of `config_hash`: the seed is a
    property of one execution, not of the configuration, and a directory name that changed
    because a hash moved cannot be audited by eye. `unseeded` is a real state — the run applied
    no seed — and is not the same as seed 0.
    """
    return "unseeded" if seed is None else f"seed{seed}"
```

Change the two methods:

```python
    def path_for(self, dataset: str, method: str, category: str, seed: int | None) -> Path:
        return self.root / f"{dataset}__{method}__{category}__{seed_tag(seed)}.parquet"

    def is_done(self, dataset: str, method: str, category: str, seed: int | None) -> bool:
        return self.path_for(dataset, method, category, seed).is_file()
```

`seed` is required on both, with no default: a caller who forgets it would otherwise silently
check the `unseeded` slot and skip a seeded category.

In `write`, take the path's seed from the meta being stamped:

```python
        final = self.path_for(dataset, method, category, meta.get("seed"))
```

with the comment:

```python
        # The seed comes from the meta this shard is about to record, so the filename and the
        # `seed` column cannot disagree. A separate seed argument would be a second value to
        # keep in step, which is the class of bug this whole change removes.
```

- [ ] **Step 4: Pass it from the runner**

In `src/vlmab/eval/runner.py`, import `seed_tag` alongside `ResultStore`, then:

```python
    todo = [c for c in wanted if not store.is_done(dataset.name, method.name, c, method.seed)]
```

and the map directory:

```python
            category_maps = (
                Path(maps_dir)
                / f"{dataset.name}__{method.name}__{run_id(meta)}__{category}"
                  f"__{seed_tag(method.seed)}"
            )
```

Extend `run_id`'s docstring with the reason the suffix sits outside it:

```
    The seed is NOT folded into this hash. It is appended to the directory name as a readable
    `seed<N>` / `unseeded` tag instead, because `config_hash` describes a configuration and the
    seed describes one execution of it — and because a directory whose name changed for a
    visible reason can be audited, while one whose hash moved cannot.
```

- [ ] **Step 5: Update the existing tests that name a shard path**

- `tests/test_store.py`: append `, 0` to the `path_for` / `is_done` calls on lines **18, 24, 49,
  51, 78, 80, 99, 119, 120**. Every one of those tests either writes with `{"seed": 0}` or writes
  nothing at all, so `0` is right throughout; the value only has to match what the test wrote.

  **`test_failed_write_preserves_previous_shard` (line ~83) needs a real fix, not an argument.**
  It writes with `{"seed": 0}` and then makes a *crashing* second write with `{"seed": 999}`.
  Those are now two different paths, so the test would still pass while no longer testing
  anything: the completed shard survives because nothing ever aimed at it. Change the crashing
  write's meta to `{"seed": 0}` so it targets the same final path again — the payload row
  (`z.png`) is already different, which is what makes "the original survived" meaningful. Confirm
  the test still fails if `write` is made to skip the tmp+rename (revert after checking).
- `tests/test_runner.py:122` — `store.path_for("mvt", "echo", "can")` gains `, None`
  (`_EchoMethod` declares no seed).
- `tests/test_runner.py:195` — the expected name becomes
  `"fake__counting__beta__unseeded.parquet"`.
- `tests/test_runner.py` orphan-map test (~line 277) — the directory becomes
  `f"fake__counting__{run_id(meta)}__alpha__unseeded"`.
- `tests/test_calibrate_threshold.py:95` — replace the hardcoded name with
  `shard = ResultStore(results).path_for("mvtec_ad2", "intensity_baseline", "vial", 0)`
  (`_validation_run` writes with `{"seed": 0}`).

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass. Then confirm nothing still calls the old three-argument form:
`grep -rn "path_for(\|is_done(" src scripts tests | grep -v "def "` and read each hit for a
fourth argument.

- [ ] **Step 7: Commit**

```bash
git add src/vlmab/eval tests/
git commit -m "fix: a seed gets its own shard path and its own map directory

Two seeds resolved to one shard path, so is_done() called the second one
finished and skipped it; and to one map directory, since run_id is
config_hash and carries no seed, so the second overwrote the first's maps
while the first's shard still pointed at them."
```

---

### Task 5: `--seed` reaches the method it seeds

**Files:**
- Modify: `src/vlmab/methods/registry.py`
- Modify: `scripts/run_eval.py:28,34-38`
- Test: `tests/test_registry.py`, `tests/test_run_eval.py`

**Interfaces:**
- Consumes: adapter constructors accepting `seed=` (Task 2).
- Produces: `build_method(name: str, seed: int | None = None) -> AnomalyMethod`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_registry.py`:

```python
def test_build_method_forwards_the_seed_to_the_adapter():
    from vlmab.methods.patchcore_ref import PatchCoreRef

    method = build_method("patchcore_ref", seed=3)
    assert isinstance(method, PatchCoreRef) and method.seed == 3


def test_build_method_lets_a_deterministic_method_refuse_a_seed():
    """The registry does not know which methods are stochastic and must not decide: it forwards,
    and the adapter that knows raises."""
    with pytest.raises(ValueError, match="deterministic"):
        build_method("intensity_baseline", seed=0)


def test_every_registered_method_builds_without_a_seed():
    for name in available():
        assert build_method(name).seed is None
```

Append to `tests/test_run_eval.py`:

```python
def test_run_eval_records_no_seed_when_none_was_asked_for(tmp_path):
    build_category(tmp_path, "vial")
    results = tmp_path / "results"
    assert main(["--method", "intensity_baseline", "--root", str(tmp_path),
                 "--split", "test_public", "--results", str(results)]) == 0
    df = ResultStore(results).load_all()
    assert "seed" in df.columns and df["seed"].isna().all()


def test_run_eval_reports_a_seed_a_deterministic_method_cannot_apply(tmp_path, capsys):
    """--seed used to be recorded and applied to nothing. Now the method that cannot apply it
    says so, and the run refuses rather than writing a shard that claims a seed."""
    build_category(tmp_path, "vial")
    code = main(["--method", "intensity_baseline", "--root", str(tmp_path),
                 "--split", "test_public", "--results", str(tmp_path / "r"), "--seed", "0"])
    assert code == 1
    assert "deterministic" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_registry.py tests/test_run_eval.py -q`
Expected: FAIL — `build_method() got an unexpected keyword argument 'seed'`, and the CLI test
seeing an uncaught `ValueError` rather than exit code 1.

- [ ] **Step 3: Forward the seed from the registry**

In `src/vlmab/methods/registry.py`:

```python
def build_method(name: str, seed: int | None = None) -> AnomalyMethod:
    """Build an adapter by name, handing it the seed the caller asked for.

    The registry does not know which methods are stochastic and must not decide: it forwards,
    and an adapter that cannot apply a seed refuses it (ValueError) rather than letting a number
    reach provenance that nothing applied.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; available: {available()}")
    return _REGISTRY[name](seed=seed)
```

The `_REGISTRY` type annotation becomes
`dict[str, Callable[..., AnomalyMethod]]`, since the factories now take a keyword argument.

- [ ] **Step 4: Wire the CLI**

In `scripts/run_eval.py`, change the argument's default and report a refusal cleanly:

```python
    parser.add_argument(
        "--seed", default=None, type=int,
        help="seed to apply to a stochastic method; omit for a method that seeds nothing "
             "(recorded as unseeded, which is the truth)",
    )
```

```python
    try:
        method = build_method(args.method, seed=args.seed)
    except KeyError as exc:
        print(exc.args[0])
        print(f"available methods: {available()}")
        return 1
    except ValueError as exc:
        # A method that cannot apply the seed it was handed. Report it the way an unknown
        # method is reported, rather than a traceback: the run is refused, not broken.
        print(f"{args.method}: {exc}")
        return 1
```

Update the module docstring's example invocation to drop `--seed 0`.

- [ ] **Step 5: Update the existing CLI tests**

Four calls pass `"--seed", "0"` — lines **13, 50, 65, 92**. They do not all need the same
treatment, and the difference is worth reading rather than sedding over:

- **Lines 13 and 50 must drop it.** Both run the real `intensity_baseline` through `main`, which
  now builds it with `seed=0` and exits 1. The seed was never applied in those runs, so removing
  it changes nothing they assert.
- **Line 65 keeps it.** That test runs `mllm_qwen`, which is not deterministic and accepts a seed;
  it now also demonstrates that the flag reaches a stochastic adapter. It still exits 1 from
  `prepare()`'s `MethodNotRunnable`, which is what the test asserts.
- **Line 92 drops it.** `build_method` is monkeypatched there, so the seed reaches a lambda that
  ignores it — leaving the flag in place would suggest a coverage the test does not have.

Both monkeypatch lambdas take one positional argument and no longer match the call:
`tests/test_run_eval.py:58` becomes `lambda name, seed=None: _PrepareForbidden()` and
`tests/test_run_eval.py:89` becomes `lambda name, seed=None: _CudaFailure()`.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/vlmab/methods/registry.py scripts/run_eval.py tests/
git commit -m "feat: --seed reaches the method instead of only the record

The flag now goes to the adapter through build_method; a method that cannot
apply it refuses, and the CLI reports that the way it reports an unknown
method. The default is None, which is what a run that seeded nothing should
say."
```

---

### Task 6: Metrics refuse a frame that pools seeds

**Files:**
- Modify: `src/vlmab/eval/aggregate.py`
- Modify: `scripts/calibrate_threshold.py`
- Test: `tests/test_aggregate.py`, `tests/test_calibrate_threshold.py`

**Interfaces:**
- Consumes: the `seed` column every shard now carries (Tasks 3-4).
- Produces: `_require_single_seed(df: pd.DataFrame) -> None` in `vlmab.eval.aggregate`, called
  from `image_metrics`, `pixel_metrics` and `threshold_metrics`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_aggregate.py`:

```python
def _two_seed_frame(builder, tmp_path):
    """One category scored under two seeds, as ResultStore.load_all() returns it."""
    a = builder(tmp_path)
    a["seed"] = 0
    b = builder(tmp_path)
    b["seed"] = 1
    return pd.concat([a, b], ignore_index=True)


def test_image_metrics_refuses_a_frame_pooling_two_seeds(tmp_path):
    """Three seeds pooled is the same image counted three times as independent samples: the
    number comes out plausible and wrong, with nothing to flag it (protocol §6)."""
    with pytest.raises(ValueError, match="pools 2 seeds"):
        image_metrics(_two_seed_frame(_shard, tmp_path))


def test_pixel_metrics_refuses_a_frame_pooling_two_seeds(tmp_path):
    with pytest.raises(ValueError, match="pools 2 seeds"):
        pixel_metrics(_two_seed_frame(_shard, tmp_path))


def test_threshold_metrics_refuses_a_frame_pooling_two_seeds(tmp_path):
    with pytest.raises(ValueError, match="pools 2 seeds"):
        threshold_metrics(
            _two_seed_frame(_threshold_shard, tmp_path), _threshold_artifact(tmp_path), "vial"
        )


def test_aggregate_refuses_a_frame_pooling_two_seeds(tmp_path):
    with pytest.raises(ValueError, match="pools 2 seeds"):
        aggregate(_two_seed_frame(_shard, tmp_path))


def test_one_seed_is_fine_and_no_seed_column_is_fine(tmp_path):
    """A real single-seed shard must pass, and so must the bare frames these tests build to
    exercise the mathematics — every frame the runner writes carries the column, and refusing
    its absence would only break the fixtures."""
    one = _shard(tmp_path)
    one["seed"] = 0
    assert image_metrics(one)["n"] == len(one)
    assert image_metrics(_shard(tmp_path))["n"] == 4
```

Append to `tests/test_calibrate_threshold.py`:

```python
def test_calibrate_refuses_shards_that_pool_two_seeds(tmp_path):
    """A threshold is fitted on one run's maps. Pooling two seeds fits it on a distribution no
    single run produced, which is not the pre-registered procedure (protocol §4)."""
    import pandas as pd

    results = _validation_run(tmp_path)
    shard = ResultStore(results).path_for("mvtec_ad2", "intensity_baseline", "vial", 0)
    other = pd.read_parquet(shard)
    other["seed"] = 1
    other.to_parquet(
        ResultStore(results).path_for("mvtec_ad2", "intensity_baseline", "vial", 1), index=False
    )
    with pytest.raises(ValueError, match="pool 2 seeds"):
        _run(["--results", str(results), "--dataset", "mvtec_ad2",
              "--method", "intensity_baseline", "--out", str(tmp_path / "a.yaml")])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_aggregate.py tests/test_calibrate_threshold.py -q`
Expected: FAIL — no exception raised; the metric functions happily return a number computed over
the doubled frame, which is the defect.

- [ ] **Step 3: Add the guard to `aggregate.py`**

Above `image_metrics`:

```python
def _require_single_seed(df: pd.DataFrame) -> None:
    """Refuse a frame that pools more than one seed.

    Protocol §6 asks for three seeds reported as mean ± std, which means metrics computed once
    per seed and combined afterwards. A frame holding three seeds scores every image three times
    and treats the copies as independent samples: I-AUROC over 3N rows is not the mean of three
    I-AUROCs, and nothing about the result looks wrong. `ResultStore.load_all()` concatenates
    every shard under a root, so a three-seed root reaches here as one frame by default — this
    is the guard that makes that safe, the same way `threshold_metrics` refuses a frame that
    pools two categories.

    A frame with no `seed` column passes: the metric tests build bare frames to exercise the
    mathematics, and every frame `run_evaluation` writes carries the column by construction.
    """
    if "seed" not in df.columns:
        return
    # NaN is how parquet round-trips an unseeded run's None, and NaN != NaN would make every
    # unseeded frame look like many distinct seeds. Fold it back to None before counting.
    seeds = sorted({None if pd.isna(s) else s for s in df["seed"].unique()}, key=str)
    if len(seeds) > 1:
        raise ValueError(
            f"this frame pools {len(seeds)} seeds ({seeds}): metrics are computed one seed at a "
            "time and combined afterwards (protocol §6), never over the pooled rows"
        )
```

Call it as the first statement of `image_metrics`, `pixel_metrics` and `threshold_metrics`.
`aggregate` needs no call of its own — it reaches both metric functions.

- [ ] **Step 4: Add the matching refusal to `calibrate_threshold.py`**

Add `import pandas as pd` to the imports, and insert after the `split` check (immediately before
the `map_path` check):

```python
    # A threshold is fitted on the maps one run produced. Two seeds pooled fits it on a mixture
    # no single run ever produces, and calibration would still succeed and write an artifact
    # that looks pre-registered. Same refusal the metric functions make.
    if "seed" in df.columns:
        seeds = sorted({None if pd.isna(s) else s for s in df["seed"].unique()}, key=str)
        if len(seeds) > 1:
            raise ValueError(
                f"the shards under {args.results} pool {len(seeds)} seeds ({seeds}): a threshold "
                "is calibrated on one run's maps, so pooling seeds fits it on a distribution no "
                "single run produced. Calibrate each seed separately (protocol §4)."
            )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_aggregate.py tests/test_calibrate_threshold.py -q`
Expected: all pass.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/vlmab/eval/aggregate.py scripts/calibrate_threshold.py tests/
git commit -m "fix: refuse to compute a metric over pooled seeds

load_all() concatenates every shard under a root, so a three-seed root
arrives as one frame and scores each image three times as independent
samples. The number would be plausible and wrong. Calibration refuses the
same way: a threshold is fitted on one run's maps."
```

---

### Task 7: The notebook, the protocol note, and the map

**Files:**
- Modify: `notebooks/patchcore_colab.ipynb` (cell 24 only)
- Modify: `docs/protocol.md` §6
- Modify: `docs/next-steps.md`

**Interfaces:**
- Consumes: everything above.
- Produces: no code interface. This is the handoff the next Colab session reads.

- [ ] **Step 1: Update notebook cell 24**

Change only the cell's source, never the notebook's cell structure — a structural change costs the
executor a full reload (reinstalling anomalib, re-downloading the weights), while a source change
costs one re-run of cell 1.1. The cell becomes:

```python
from vlmab.datasets.mvtec_ad2 import MVTecAD2
from vlmab.eval.runner import run_evaluation
from vlmab.eval.store import ResultStore
from vlmab.eval.provenance import run_meta
from vlmab.methods.patchcore_ref import PatchCoreRef
from vlmab.methods.patchcore_backend import PatchCoreBackend

# The seed goes on the BACKEND, which is what calls seed_everything. PatchCoreRef reads it
# from there and the runner stamps it into the shard, so the record cannot disagree with the
# run. run_meta no longer takes a seed for exactly that reason.
method = PatchCoreRef(backend=PatchCoreBackend(seed=0))
store = ResultStore("results/patchcore/vial/shards")
meta = run_meta({"method": "patchcore_ref", "split": "test_public"})

run_evaluation(
    MVTecAD2("data/mvtec_ad2"), method, store, meta,
    categories=["vial"], split="test_public",
    maps_dir="results/patchcore/vial/maps",
)
print("declared seed:", method.seed)
```

Edit it with a script rather than by hand so the rest of the JSON is untouched. The file is written
with `indent=1`, which is what nbformat produces, so rewriting it this way changes only the one
cell:

```bash
.venv/bin/python - <<'PY'
import json

SOURCE = '''from vlmab.datasets.mvtec_ad2 import MVTecAD2
from vlmab.eval.runner import run_evaluation
from vlmab.eval.store import ResultStore
from vlmab.eval.provenance import run_meta
from vlmab.methods.patchcore_ref import PatchCoreRef
from vlmab.methods.patchcore_backend import PatchCoreBackend

# The seed goes on the BACKEND, which is what calls seed_everything. PatchCoreRef reads it
# from there and the runner stamps it into the shard, so the record cannot disagree with the
# run. run_meta no longer takes a seed for exactly that reason.
method = PatchCoreRef(backend=PatchCoreBackend(seed=0))
store = ResultStore("results/patchcore/vial/shards")
meta = run_meta({"method": "patchcore_ref", "split": "test_public"})

run_evaluation(
    MVTecAD2("data/mvtec_ad2"), method, store, meta,
    categories=["vial"], split="test_public",
    maps_dir="results/patchcore/vial/maps",
)
print("declared seed:", method.seed)'''

path = "notebooks/patchcore_colab.ipynb"
nb = json.load(open(path))
cell = nb["cells"][24]
assert "run_evaluation(" in "".join(cell["source"]), "cell 24 is not the phase-2 run cell"
cell["source"] = SOURCE.splitlines(keepends=True)
cell["outputs"] = []
cell["execution_count"] = None
with open(path, "w") as fh:
    json.dump(nb, fh, indent=1)
    fh.write("\n")
PY
```

Then confirm the change is confined to that cell: `git diff --stat notebooks/patchcore_colab.ipynb`
should report a handful of changed lines, not the whole file. If it reports the whole file, the
round-trip reformatted something — revert and edit the JSON in place instead.

- [ ] **Step 2: Add the sentence to protocol §6**

In `docs/protocol.md`, after "Three seeds where any stochasticity exists; report mean ± std.":

```
Each seed is a separate run with its own shard (`…__seed<N>.parquet`) and its own anomaly-map
directory; the seed recorded in a shard is the seed the method reports having applied, and the
metric functions refuse a frame that pools seeds. A run that applied no seed records `unseeded`,
which is not the same as seed 0.
```

**Do not add a Changelog entry.** This describes the implementation of a rule §6 already fixed;
the protocol itself is unchanged and stays at v0.2.11.

- [ ] **Step 3: Update `docs/next-steps.md`**

Replace the section "PatchCore is stochastic, and `--seed` is not wired to anything (2026-08-26)"
open item with the resolution. The section's **Closed** paragraph about `PatchCoreBackend(seed=)`
stays; the **Open — a decision, not a task** paragraph is replaced by:

```markdown
**Closed 2026-09-07.** A method now declares the seed it applied (`AnomalyMethod.seed`), and
`run_evaluation` is the only writer of that field — `run_meta` no longer accepts one, so neither
entry point can record a seed nothing applied. Two further defects were found while fixing it and
are fixed with it: two seeds resolved to one shard path (so `is_done()` called the second one
finished and skipped it) and to one anomaly-map directory (`run_id` is `config_hash`, which carries
no seed, so the second overwrote the first's maps while the first's shard still pointed at them).
Both now carry a literal `seed<N>` / `unseeded` tag. The metric functions and
`scripts/calibrate_threshold.py` refuse a frame that pools seeds.

Still not built, deliberately: combining three seeds into mean ± std. That is reporting, it belongs
with `scripts/make_tables.py` (a `TODO(M5)` stub), and doing it wrong now raises rather than
publishing. Spec: `docs/superpowers/specs/2026-09-07-seed-provenance-design.md`.
```

In "Next steps, in order", item 1.3's warning — "⚠️ Cell 24 builds `PatchCoreBackend()` with no seed
while `run_meta(..., seed=0)` writes `seed: 0` into the shard" — is now false. Replace it with:
"Cell 24 seeds the backend (`PatchCoreBackend(seed=0)`) and the runner stamps that seed; phase 2's
shard is still a plumbing check and must not be reported."

In item 1.4, drop "but settle the seed wiring before that gate runs" from the phase-3 line and the
matching sentence in the 2026-08-26 handoff section, replacing them with a pointer to the closed
section above.

- [ ] **Step 4: Run the whole suite one last time**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add notebooks/patchcore_colab.ipynb docs/protocol.md docs/next-steps.md
git commit -m "docs: the seed is wired; cell 24 seeds the backend

Cell 24 changes source only, never structure: a structural change costs the
executor a reinstall and a re-download, a source change costs one re-run of
cell 1.1."
```

- [ ] **Step 6: Merge**

```bash
git checkout master
git merge --no-ff plan-seed-provenance
.venv/bin/python -m pytest -q
git push origin master
```

Push matters beyond the usual reason: a live Colab session picks fixes up by hard-resetting to
`origin/master` in cell 1.1, so an unpushed fix does not exist as far as the next session is
concerned.

---

## Notes for the executor

**One thing this plan found that the spec does not cover.** `scripts/calibrate_threshold.py` reads
shards through `ResultStore.load_all()`, which pools every seed under a root, and it does not go
through the metric functions — so the spec's guard would have missed it. Task 6 adds the matching
refusal there. If you find another `load_all()` consumer while working, it needs the same treatment;
`grep -rn "load_all()" src scripts` is the check.

**`notebooks/probe_patchcore.py` needs no change** — checked. Its stage 12 constructs
`PatchCoreBackend(seed=seed)` through the constructor keyword, which this plan does not touch, and it
never reads the private `_seed` that Task 2 renames.

**Expected total:** 313 existing tests plus roughly 30 new ones, all on CPU. No task in this plan
needs a GPU, anomalib, or a dataset download — which is the point: it clears the last blocker in
front of the PatchCore VisA gate without spending a Colab session on it.
