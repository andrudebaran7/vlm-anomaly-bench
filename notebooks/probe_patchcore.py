"""Stage-by-stage probe for the PatchCore backend.

Why this exists: `PatchCoreBackend.fit()` does five distinct things behind one call, so when it
fails the traceback lands six frames deep in Lightning and says nothing about WHICH of our
assumptions broke. Three fixes were made by inferring from tracebacks, and each one surfaced a
new failure in the same constructor -- the signal that guessing has stopped working.

Each probe below states what it EXPECTS, runs the smallest thing that tests it, and reports
against that expectation. Execution stops at the first failure with the state that matters
dumped, so the next question is always "why did stage N fail", never "where in Lightning are we".

Usage in Colab:

    from probe_patchcore import run
    ctx = run()                 # stops at the first failure
    ctx = run(keep_going=True)  # runs every stage, for a full picture

`only=` runs a single stage, but the stages share one `ctx` and later ones depend on what
earlier ones put there (the model, the transform, the images), so it is for re-running a stage
inside a session that already got that far -- not for jumping straight to stage 10 from cold.

Nothing here imports anomalib at module level, so importing this file is always safe.
"""
import traceback
from typing import Any, Callable

_STAGES: list[tuple[str, str, Callable]] = []


def stage(name: str, expect: str):
    """Register a probe. The `expect` string is the whole point: it is the claim being tested,
    written down before the result is known, so a surprising PASS is as visible as a FAIL."""
    def deco(fn):
        _STAGES.append((name, expect, fn))
        return fn
    return deco


def run(keep_going: bool = False, only: str | None = None) -> dict[str, Any]:
    """Run the probes in order, threading a shared `ctx` dict through them."""
    ctx: dict[str, Any] = {}
    width = max(len(n) for n, _, _ in _STAGES)
    for name, expect, fn in _STAGES:
        if only and only != name:
            continue
        print(f"\n{'=' * 72}\n{name.ljust(width)}  EXPECT: {expect}\n{'-' * 72}")
        try:
            note = fn(ctx)
            print(f"PASS  {note or ''}")
        except Exception as exc:
            print(f"FAIL  {type(exc).__name__}: {exc}\n")
            traceback.print_exc()
            ctx["_failed_at"] = name
            if not keep_going:
                print(f"\n{'=' * 72}\nStopped at: {name}\n"
                      f"Everything before it holds. Fix this stage's assumption, not a later one.")
                return ctx
    ctx.setdefault("_failed_at", None)
    return ctx


# --------------------------------------------------------------------------------------------
# Stages. Order matters: each one may only depend on what earlier stages put in ctx.
# --------------------------------------------------------------------------------------------

@stage("0. environment", "anomalib and torch import; a CUDA device is visible")
def _env(ctx):
    import anomalib, torch
    ctx["anomalib_version"] = anomalib.__version__
    ctx["cuda"] = torch.cuda.is_available()
    assert ctx["cuda"], "no CUDA device -- the backend cannot run"
    return f"anomalib {anomalib.__version__}, torch {torch.__version__}"


@stage("1. signatures", "we can enumerate what Folder/Engine/Patchcore ACCEPT, before passing anything")
def _signatures(ctx):
    import inspect
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    for label, obj in (("Folder", Folder), ("Engine", Engine), ("Patchcore", Patchcore)):
        sig = inspect.signature(obj.__init__)
        params = [p for p in sig.parameters if p != "self"]
        var_kw = [p.name for p in sig.parameters.values() if p.kind == p.VAR_KEYWORD]
        ctx[f"{label}_params"] = params
        ctx[f"{label}_varkw"] = bool(var_kw)
        print(f"  {label}: {params}")
        if var_kw:
            # This is the trap that cost this session two rounds: a **kwargs sink accepts an
            # unknown kwarg at construction and defers the failure to whoever consumes it.
            print(f"    ** has {var_kw} -- unknown kwargs are ACCEPTED here and fail later, "
                  f"not at the call you wrote")
    return "signatures listed above; compare them against what the backend passes"


@stage("2. materialise", "20 synthetic normal images land as .png files in a temp dir")
def _materialise(ctx):
    import tempfile
    from pathlib import Path
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    images = [rng.integers(90, 110, (256, 256, 3), dtype=np.uint8) for _ in range(20)]
    ctx["images"] = images
    ctx["anom"] = images[0].copy()
    ctx["anom"][40:80, 40:80] = 255

    ctx["tmp"] = tempfile.TemporaryDirectory()
    good = Path(ctx["tmp"].name) / "good"
    good.mkdir(parents=True)
    for i, img in enumerate(images):
        Image.fromarray(img).save(good / f"{i:05d}.png")

    n = len(list(good.glob("*.png")))
    assert n == 20, f"wrote {n} files, expected 20"
    ctx["root"] = ctx["tmp"].name
    return f"{n} pngs under {good}"


@stage("3. Folder ctor", "Folder builds from normal-only images with both splits disabled")
def _folder(ctx):
    from anomalib.data import Folder
    from anomalib.data.utils import TestSplitMode, ValSplitMode

    # num_workers=2: Colab has 2 CPUs, anomalib defaults to 8 and the DataLoader warns twice
    # per setup about it. Same value the backend uses.
    kwargs = dict(name="fit", root=ctx["root"], normal_dir="good", num_workers=2,
                  val_split_mode=ValSplitMode.NONE, test_split_mode=TestSplitMode.NONE)
    unknown = [k for k in kwargs if k not in ctx.get("Folder_params", kwargs)]
    assert not unknown, f"Folder does not accept {unknown}"
    ctx["datamodule"] = Folder(**kwargs)
    return f"built with {sorted(kwargs)}"


@stage("4. datamodule.setup", "setup() succeeds and the train dataloader yields the 20 images")
def _setup(ctx):
    dm = ctx["datamodule"]
    dm.setup()
    dl = dm.train_dataloader()
    n = len(dl.dataset)
    assert n == 20, f"train dataset has {n} items, expected 20"
    batch = next(iter(dl))
    ctx["batch_type"] = type(batch).__name__
    ctx["batch_fields"] = [f for f in dir(batch) if not f.startswith("_")][:12] \
        if not isinstance(batch, (list, tuple, dict)) else batch
    print(f"  batch type: {ctx['batch_type']}")
    return f"{n} train items, one batch pulled"


#: The Engine configuration verified on anomalib 2.6.0 / Colab T4 (2026-08-21) by running
#: four candidates and measuring which filled the coreset. Mirrors
#: src/vlmab/methods/patchcore_backend.py -- stage 5 asserts the two have not drifted apart.
VERIFIED_ENGINE_KWARGS = dict(
    logger=False,
    enable_progress_bar=False,   # without this, rich + anomalib's tqdm nest until RecursionError
    limit_val_batches=0,         # ValSplitMode.NONE means val_data never exists
    num_sanity_val_steps=0,
)


@stage("5. Engine ctor", "Engine builds with the verified kwargs, and the backend still uses them")
def _engine(ctx):
    import inspect
    from anomalib.engine import Engine

    # These go through Engine's **kwargs sink to Lightning's Trainer on purpose: they are
    # Trainer arguments, and this exact set is the one measured to work. Passing an argument
    # Trainer does not accept would still fail one call later, which is why the set is fixed
    # rather than assembled.
    ctx["engine"] = Engine(**VERIFIED_ENGINE_KWARGS)
    ctx["engine_kwargs"] = dict(VERIFIED_ENGINE_KWARGS)

    # Drift guard: the probe exists to validate what ships, so if the backend's Engine call
    # stops matching this set, say so here rather than silently testing something else.
    try:
        from vlmab.methods.patchcore_backend import PatchCoreBackend
        src = inspect.getsource(PatchCoreBackend.__init__)
        missing = [k for k in VERIFIED_ENGINE_KWARGS if f"{k}=" not in src]
        if missing:
            print(f"  WARNING: backend's Engine call is missing {missing} -- probe and backend "
                  f"have drifted, and this stage is no longer testing what ships")
    except Exception as exc:   # the backend is not importable in every environment
        print(f"  (could not cross-check the backend: {type(exc).__name__}: {exc})")

    return f"built with {sorted(VERIFIED_ENGINE_KWARGS)}"


@stage("6. model ctor", "Patchcore builds with the pre-registered hyperparameters")
def _model(ctx):
    from anomalib.models import Patchcore

    ctx["model"] = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                             coreset_sampling_ratio=0.1, num_neighbors=9)
    ctx["transform"] = type(ctx["model"]).configure_pre_processor().transform
    print(f"  transform: {ctx['transform']}")
    return "model + pre-processor transform ready"


@stage("7. engine.train", "one epoch fills the coreset memory bank without a Trainer misconfiguration")
def _train(ctx):
    import torch

    ctx["engine"].train(model=ctx["model"], datamodule=ctx["datamodule"])

    # Lightning owns device placement during fit and hands the model back on CPU. Every stage
    # after this one calls the model directly with a CUDA tensor, so reclaim it here -- the same
    # two lines PatchCoreBackend.fit() ends with. Without them stage 8 fails with
    # "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same",
    # which says nothing about training and everything about the seam between the two ways the
    # model is driven.
    ctx["model"].to("cuda")
    ctx["model"].eval()
    inner = getattr(ctx["model"], "model", None)
    bank = getattr(inner, "memory_bank", None)
    if isinstance(bank, torch.Tensor) and bank.numel():
        if bank.device.type != "cuda":
            inner.memory_bank = bank.to("cuda")
        ctx["memory_bank_shape"] = tuple(bank.shape)
        print(f"  memory bank: {tuple(bank.shape)} on {inner.memory_bank.device}")
    return "trained; memory bank filled"


def _model_input(ctx, img):
    """Prepare one image the way PatchCoreBackend.score() does, and for the same reason.

    The pre-processor is `Resize([256,256]) + Normalize(imagenet)` with **no ToTensor**, so it
    rejects a PIL image outright ("Normalize() does not support PIL images"). The conversion is
    ours to do: HWC uint8 -> CHW float in [0,1], the range those ImageNet statistics are defined
    against. Stages 8-10 all go through here so the probe exercises the shipped path rather than
    a plausible-looking one -- this file has already been wrong that way once (see stage 5's
    drift guard).
    """
    import numpy as np
    import torch

    arr = np.asarray(img, dtype=np.uint8)
    base = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
    return ctx["transform"](base).unsqueeze(0).to("cuda")


@stage("8. forward shape", "a forward pass returns something exposing pred_score and anomaly_map")
def _forward(ctx):
    import torch

    t = _model_input(ctx, ctx["images"][1])
    with torch.no_grad():
        out = ctx["model"](t)
    ctx["out_type"] = type(out).__name__
    attrs = [a for a in ("pred_score", "anomaly_map", "pred_label", "pred_mask") if hasattr(out, a)]
    print(f"  returned {ctx['out_type']}, has: {attrs}")
    assert "pred_score" in attrs and "anomaly_map" in attrs, \
        f"{ctx['out_type']} exposes {attrs}; score() reads pred_score and anomaly_map"
    ctx["out"] = out
    return f"{ctx['out_type']} with pred_score + anomaly_map"


@stage("9. separation", "the injected bright patch scores ABOVE a normal image")
def _separation(ctx):
    import torch

    def score(img):
        t = _model_input(ctx, img)
        with torch.no_grad():
            o = ctx["model"](t)
        return float(o.pred_score.reshape(-1)[0].item())

    s_norm, s_anom = score(ctx["images"][1]), score(ctx["anom"])
    ctx["s_normal"], ctx["s_anom"] = s_norm, s_anom
    assert s_anom > s_norm, f"anomalous {s_anom} did not exceed normal {s_norm}"
    return f"normal {s_norm:.4f} < anomalous {s_anom:.4f}"


# --------------------------------------------------------------------------------------------
# Hypothesis testing for stage 7.
#
# Evidence (2026-08-21, anomalib 2.6.0, Colab T4): stage 7 fails with
#     AttributeError: 'Folder' object has no attribute 'val_data'
# raised from lightning/pytorch/loops/evaluation_loop.py setup_data -> val_dataloader.
#
# Root cause: val_split_mode=ValSplitMode.NONE means the datamodule never builds `val_data`,
# but Lightning's fit loop unconditionally sets up the validation loop before the first epoch.
# The two options are incompatible; nothing we pass to the *constructor* changes that.
#
# Note what stage 7's own log said: "configure_optimizers returned None, this fit will run with
# no optimizer". PatchCore does not train. It runs a forward pass over normal images and
# subsamples a coreset. The whole Trainer is scaffolding around that.
#
# `diagnose_train()` tests each candidate independently, on a fresh model and datamodule, and
# reports which ones fill the memory bank. One run, three answers.
# --------------------------------------------------------------------------------------------

def _fresh(root, val_mode=None, val_ratio=None, num_workers=2):
    """A new model + datamodule, so a failed candidate cannot contaminate the next."""
    from anomalib.data import Folder
    from anomalib.data.utils import TestSplitMode, ValSplitMode
    from anomalib.models import Patchcore

    kwargs = dict(name="fit", root=root, normal_dir="good",
                  test_split_mode=TestSplitMode.NONE, num_workers=num_workers)
    kwargs["val_split_mode"] = ValSplitMode.NONE if val_mode is None else val_mode
    if val_ratio is not None:
        kwargs["val_split_ratio"] = val_ratio
    dm = Folder(**kwargs)
    dm.setup()
    model = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                      coreset_sampling_ratio=0.1, num_neighbors=9)
    return dm, model


def _bank_size(model):
    """How many patches ended up in the coreset. 0 or missing means fit did nothing."""
    bank = getattr(getattr(model, "model", None), "memory_bank", None)
    if bank is None:
        return None
    try:
        return int(bank.shape[0])
    except Exception:
        return f"present, shape unavailable ({type(bank).__name__})"


def diagnose_train(root=None):
    """Try each candidate fix for stage 7. Returns {label: (ok, detail)}."""
    import tempfile, traceback
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from anomalib.data.utils import ValSplitMode
    from anomalib.engine import Engine

    if root is None:  # self-contained: build the same 20 normal images
        tmp = tempfile.TemporaryDirectory()
        good = Path(tmp.name) / "good"
        good.mkdir(parents=True)
        rng = np.random.default_rng(0)
        for i in range(20):
            Image.fromarray(rng.integers(90, 110, (256, 256, 3), dtype=np.uint8)).save(
                good / f"{i:05d}.png")
        root = tmp.name
        diagnose_train._tmp = tmp   # keep it alive

    def _engine(**extra):
        """Engine with the display layer off.

        Evidence (2026-08-21): candidates A/B/C all got PAST the val_data error and reached
        "Selecting Coreset Indices" -- the actual work -- then died in rich/console.py with
        RecursionError while the tqdm bar reprinted ~85 times without advancing. Lightning's
        RichProgressBar holds a live display and anomalib's coreset sampler writes its own tqdm
        into it; in a notebook the two nest until the recursion limit blows.
        So this is the DISPLAY layer failing, not the Trainer path being wrong.
        """
        kwargs = dict(logger=False, enable_progress_bar=False,
                      limit_val_batches=0, num_sanity_val_steps=0)
        kwargs.update(extra)
        return Engine(**kwargs)

    def candidate_a():
        """The val fix alone -- kept so the summary shows it still reaches the coreset step."""
        dm, model = _fresh(root)
        Engine(logger=False, limit_val_batches=0, num_sanity_val_steps=0).train(
            model=model, datamodule=dm)
        return model

    def candidate_b():
        """Val fix + Lightning's progress bar off. The hypothesis this round tests."""
        dm, model = _fresh(root)
        _engine().train(model=model, datamodule=dm)
        return model

    def candidate_c():
        """As B, and also silence anomalib's own tqdm, in case disabling Lightning's bar is
        not enough on its own."""
        import os
        os.environ["TQDM_DISABLE"] = "1"
        try:
            dm, model = _fresh(root)
            _engine().train(model=model, datamodule=dm)
            return model
        finally:
            os.environ.pop("TQDM_DISABLE", None)

    def candidate_d():
        """Diagnostic, not a fix: raise the recursion limit. If this alone gets through, the
        nesting is deep-but-finite; if it still blows, the rendering recurses without bound and
        no limit will save it. Either answer is informative."""
        import sys as _sys
        old = _sys.getrecursionlimit()
        _sys.setrecursionlimit(20000)
        try:
            dm, model = _fresh(root)
            Engine(logger=False, limit_val_batches=0, num_sanity_val_steps=0).train(
                model=model, datamodule=dm)
            return model
        finally:
            _sys.setrecursionlimit(old)

    results = {}
    for label, fn in (("A  val fix only (baseline)", candidate_a),
                      ("B  + enable_progress_bar=False", candidate_b),
                      ("C  + TQDM_DISABLE too", candidate_c),
                      ("D  raised recursion limit", candidate_d)):
        print(f"\n{'=' * 72}\n{label}\n{'-' * 72}")
        try:
            model = fn()
            n = _bank_size(model)
            ok = n not in (None, 0)
            print(f"{'PASS' if ok else 'FAIL'}  memory bank: {n}")
            results[label] = (ok, f"memory bank {n}")
        except Exception as exc:
            print(f"FAIL  {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
            results[label] = (False, f"{type(exc).__name__}: {exc}")

    print(f"\n{'=' * 72}\nSUMMARY")
    for label, (ok, detail) in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {label:32} {detail}")
    winners = [l for l, (ok, _) in results.items() if ok]
    print(f"\nUsable: {winners or 'none -- the Trainer path may be the wrong abstraction'}")
    return results


@stage("10. double preprocessing",
       "the model does NOT re-apply the pre-processor, so applying it once in score() is right")
def _double_preprocess(ctx):
    """The dangerous case, because it fails silently rather than raising.

    `PreProcessor` is a module INSIDE the Patchcore model (it shows up in the Lightning
    summary as child 0). If the forward pass applies it as well as our score() doing so, the
    image is resized and ImageNet-normalised twice: no error, just wrong numbers, and the VisA
    gate would fail for a reason no traceback would ever point at.

    Test: feed the same image raw ([0,1], resized only) and pre-processed (normalised), and
    compare. If the model normalised internally, the raw input is the correctly-prepared one
    and the two scores differ in a specific direction. Whatever the outcome, it is measured.
    """
    import torch
    import numpy as np
    from torchvision.transforms.v2 import Resize

    img = np.asarray(ctx["anom"], dtype=np.uint8)
    base = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)

    resized_only = Resize([256, 256], antialias=True)(base).unsqueeze(0).to("cuda")
    preprocessed = _model_input(ctx, img)   # exactly what score() sends

    with torch.no_grad():
        s_raw = float(ctx["model"](resized_only).pred_score.reshape(-1)[0].item())
        s_pre = float(ctx["model"](preprocessed).pred_score.reshape(-1)[0].item())

    ctx["score_raw_input"], ctx["score_preprocessed_input"] = s_raw, s_pre
    print(f"  raw [0,1] resized only : {s_raw:.6f}")
    print(f"  pre-processed (ours)   : {s_pre:.6f}")
    print(f"  ratio                  : {s_pre / s_raw if s_raw else float('nan'):.4f}")
    if abs(s_raw - s_pre) < 1e-9:
        return ("identical scores -> the model normalises internally and IGNORES what we pass; "
                "score() must NOT pre-process")
    return ("scores differ -> the model consumes what it is given, so score() applying the "
            "pre-processor once is correct")
