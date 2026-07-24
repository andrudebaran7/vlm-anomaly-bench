# PatchCore anomalib backend — Colab integration

The `PatchCoreRef` adapter (`src/vlmab/methods/patchcore_ref.py`) wraps an injected backend so its
full-shot orchestration is tested on CPU. The real backend runs anomalib on a GPU and is built in a
Colab session, not in this repository's tests — anomalib's API differs across the pinned range and
cannot be exercised without a GPU.

## The backend contract

Inject any object with these two methods into `PatchCoreRef(backend=...)`:

    fit(train_images: Iterable[np.ndarray]) -> None
        Build the coreset memory bank from a category's defect-free training images (HxWx3 uint8,
        as the loader yields them). Called once per category by the runner.

    score(image: np.ndarray) -> tuple[float, np.ndarray]
        Return (raw image-level score, raw anomaly map). The map may be at any resolution; the
        adapter upsamples it to the image's native resolution. Neither value is per-image
        normalised (protocol v0.2.6) — return anomalib's raw scores.

## Hyperparameters

Frozen in `configs/methods/patchcore_ref.yaml`: `wide_resnet50_2`, layers `layer2`+`layer3`,
coreset ratio 0.1, 9 neighbours. Record the resolved anomalib version there on the first run.

## Validation gate (protocol §2)

Before any MVTec AD 2 number is reported, the backend must reproduce PatchCore's published VisA
image-AUROC within ±1.0 point, run through this same adapter and the project's own metrics. A
failure to reproduce flags the anchor in every table.

## Running it

Once the backend is built and injected, everything else is already wired:
`scripts/run_eval.py --method patchcore_ref ...` fits per category and scores through the runner,
exactly as `intensity_baseline` does today. Registered with no backend, `patchcore_ref` reports
cleanly via `MethodNotRunnable` (a `prepare()` on CPU exits 1), so the CLI never pretends it ran.
