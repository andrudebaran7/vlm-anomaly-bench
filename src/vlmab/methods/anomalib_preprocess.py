"""Pre-processing helper shared by the anomalib backends (no torch or anomalib import here).

anomalib 2.6.0's `PreProcessor.forward` -- the path a direct model call takes -- applies
`export_transform`, a copy of `transform` with antialiasing disabled on every Resize (an ONNX-export
accommodation), while its Lightning test loop and the fit path apply `transform` itself, antialias
on. WinCLIP (protocol §3 v0.2.20) and PatchCore (§3 v0.2.21) both route forward to the antialiased
transform; each passes its own protocol entry so a refusal points at the right decision.
"""


def use_antialiased_forward(pre, protocol_ref: str) -> None:
    """Make `pre.forward` apply `pre.transform` (antialiased) instead of `pre.export_transform`.

    `pre` is the model's anomalib PreProcessor (a parameter so CPU tests can pass a stand-in).
    `forward` reads `self.export_transform` at call time, so pointing it at `transform` is enough;
    no anomalib code is replaced. Refuses unless `transform` really contains an antialiasing
    Resize, so the configuration recorded under `protocol_ref` cannot be claimed for a pipeline
    without it.
    """
    transform = getattr(pre, "transform", None)
    steps = list(getattr(transform, "transforms", [transform]))
    resizes = [t for t in steps if type(t).__name__ == "Resize"]
    if not resizes:
        raise RuntimeError(
            f"the pre-processor's transform {steps!r} has no Resize; {protocol_ref} assumed "
            "anomalib's antialiasing Resize"
        )
    if not all(getattr(r, "antialias", False) is True for r in resizes):
        raise RuntimeError(
            f"the pre-processor's Resize does not antialias ({resizes!r}); routing forward to it "
            f"would not give the antialiased resize {protocol_ref} records"
        )
    pre.export_transform = transform
