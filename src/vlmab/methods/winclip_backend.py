"""anomalib WinCLIP backend for WinClipRef (GPU/Colab only).

Bridges the seam score(image, category) -> (raw_score, raw_map) to anomalib 2.6.0's zero-shot
WinClip. Four things here are not what anomalib does on its own, and each is deliberate:

1. **The prompt templates are the paper's 22, not anomalib's 21** (protocol §3 v0.2.19).
   anomalib's `prompting.TEMPLATES` lacks "a cropped photo of a {}." and "a jpeg corrupted photo
   of a {}." and lists "a jpeg corrupted photo of the {}." twice, which double-weights it in the
   per-class mean: 147/84 prompts against the paper's 154/88 (Fig. 6, arXiv:2303.14814; read
   from the wheel and observed on Colab 2026-10-03). The state words already match and are
   untouched.

2. **The text embeddings are built explicitly, per category.** `WinClip(class_name=...)` stores
   the name but constructs `WinClipModel()` WITHOUT it — observed 2026-10-03: `m.model.class_name`
   is None after construction and `text_embeddings` raises until `setup` runs. The class name is
   the prompt noun, so a category change must rebuild them.

3. **No post-processor.** `AnomalibModule.forward` runs pre_processor -> model -> post_processor,
   and the PostProcessor min-max normalises scores once it has statistics. With none (no
   validation run) its `_normalize` is a no-op, so the scores would be raw today — by accident of
   state rather than by construction. `post_processor=False` makes them raw by construction
   (protocol v0.2.6: no per-image normalisation; WinClipRef upsamples the map to native size).

4. **forward() resizes WITH antialiasing** (protocol §3 v0.2.20). `PreProcessor.forward` applies
   `export_transform`, a copy of `transform` with antialiasing disabled on every Resize (an ONNX
   accommodation), while anomalib's own Lightning test loop and the paper's PIL bicubic both
   antialias. Measured 2026-10-03, notebook cell 2.2: forward == the export path to 8 decimals,
   0.013 away from the antialiased one. `use_antialiased_forward` routes forward to `transform`.

The PRE-processor is kept, and that is why `score()` hands the model a raw [0,1] tensor at native
resolution: forward applies Resize([240, 240], BICUBIC) + CLIP Normalize itself. Doing either here
would apply it twice — PatchCore's backend did exactly that for weeks without a failing test.

anomalib, torch and CLIP are imported lazily so this module imports in CI (never constructed there).
"""
import numpy as np

#: The paper's 22 templates, verbatim from Figure 6 (c) of arXiv:2303.14814v1 (read from the PDF
#: 2026-10-03; full list in ../vlm-anomaly-paper/docs/verified-literature-facts.md), in the
#: paper's order, with anomalib's "{}" in place of the paper's "[c]".
PAPER_TEMPLATES: tuple[str, ...] = (
    "a cropped photo of the {}.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of the {}.",
    "a dark photo of a {}.",
    "a jpeg corrupted photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of the {}.",
    "a blurry photo of a {}.",
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of the {} for visual inspection.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for anomaly detection.",
    "a photo of a {} for anomaly detection.",
)

#: 7 normal and 4 anomaly state words x 22 templates (paper Fig. 6).
PAPER_PROMPT_COUNTS = {"normal": 154, "anomaly": 88}


def apply_paper_templates(prompting) -> dict[str, int]:
    """Replace `prompting.TEMPLATES` with the paper's 22, IN PLACE, and prove the result is 154/88.

    `prompting` is anomalib's `anomalib.models.image.winclip.prompting` module (a parameter so CPU
    tests can pass a stand-in). Its `create_prompt_ensemble` reads the module-level list at call
    time, so the patch reaches every text embedding built afterwards. Slice assignment rather
    than rebinding, so anything already holding the list object sees the same contents.

    Process-global by nature: every WinClip built in this interpreter after the call uses the
    paper's templates. That is the intent — no WinCLIP number in this study uses anomalib's.

    Raises RuntimeError if the ensemble is not 154/88 afterwards, i.e. if the state words are no
    longer the paper's. Those are left as anomalib ships them on the strength of a reading that
    they match; a build where they do not must stop here rather than score a different ensemble.
    """
    prompting.TEMPLATES[:] = list(PAPER_TEMPLATES)
    normal, anomalous = prompting.create_prompt_ensemble("object")
    counts = {"normal": len(normal), "anomaly": len(anomalous)}
    distinct = {"normal": len(set(normal)), "anomaly": len(set(anomalous))}
    if counts != PAPER_PROMPT_COUNTS or distinct != PAPER_PROMPT_COUNTS:
        raise RuntimeError(
            f"after installing the paper's 22 templates the ensemble is {counts} "
            f"({distinct} distinct), not the paper's 154 normal / 88 anomaly. The state words "
            f"are no longer the paper's 7 + 4 (normal={len(prompting.NORMAL_STATES)}, "
            f"anomaly={len(prompting.ANOMALOUS_STATES)}); protocol §3 v0.2.19 assumed they were."
        )
    return counts


def use_antialiased_forward(pre) -> None:
    """Make `pre.forward` apply `pre.transform` (antialiased) instead of `pre.export_transform`.

    `pre` is the model's anomalib PreProcessor (a parameter so CPU tests can pass a stand-in).
    `forward` reads `self.export_transform` at call time, so pointing it at `transform` is enough;
    no anomalib code is replaced. Refuses unless `transform` really contains an antialiasing
    Resize, so the configuration recorded in v0.2.20 cannot be claimed for a pipeline without it.
    """
    transform = getattr(pre, "transform", None)
    steps = list(getattr(transform, "transforms", [transform]))
    resizes = [t for t in steps if type(t).__name__ == "Resize"]
    if not resizes:
        raise RuntimeError(
            f"the pre-processor's transform {steps!r} has no Resize; protocol §3 v0.2.20 assumed "
            "anomalib's Resize([240, 240], BICUBIC, antialias=True)"
        )
    if not all(getattr(r, "antialias", False) is True for r in resizes):
        raise RuntimeError(
            f"the pre-processor's Resize does not antialias ({resizes!r}); routing forward to it "
            "would not give the antialiased resize protocol §3 v0.2.20 records"
        )
    pre.export_transform = transform


class WinClipBackend:
    def __init__(
        self,
        k_shot: int = 0,
        scales: tuple[int, ...] = (2, 3),
        device: str = "cuda",
        seed: int | None = None,
    ):
        from anomalib.models import WinClip  # lazy: GPU-only
        from anomalib.models.image.winclip import prompting

        if k_shot != 0:
            raise ValueError("WinClipBackend is zero-shot only; WinCLIP+ (k_shot > 0) is out of scope")

        # BEFORE any model exists, so no text embedding is ever built from anomalib's 21.
        self.prompt_counts = apply_paper_templates(prompting)

        # Public: WinClipRef reads it to declare the seed in every shard's provenance. Zero-shot
        # WinCLIP may well be deterministic, but that is a measurement owed (plan phase A.4),
        # not an assumption, so the seed is applied when given, exactly as for PatchCore.
        self.seed = seed
        if seed is not None:
            from lightning import seed_everything
            seed_everything(seed, workers=True)

        self._device = device
        self._model = WinClip(k_shot=0, scales=tuple(scales), post_processor=False)
        use_antialiased_forward(self._model.pre_processor)
        # Public, like `seed`: what a report needs to state about the input pipeline.
        self.resize_antialias = True
        self._model.eval().to(device)
        self._category: str | None = None

    def _use_category(self, category: str) -> None:
        """Build the text embeddings for `category`'s noun, once per change of category.

        `WinClipModel.setup(class_name)` runs `_collect_text_embeddings`, which places them on the
        device of the model's own parameters — so this is called after `.to(device)`, never before.
        The Lightning-level `class_name` is kept in step only so nothing reading it is misled.
        """
        if category == self._category:
            return
        inner = self._model.model
        inner.setup(category)
        if inner.class_name != category or tuple(inner.text_embeddings.shape)[0] != 2:
            raise RuntimeError(
                f"text embeddings for {category!r} were not built: class_name="
                f"{inner.class_name!r}, text_embeddings {tuple(inner.text_embeddings.shape)}"
            )
        self._model.class_name = category
        self._category = category

    def score(self, image: np.ndarray, category: str) -> tuple[float, np.ndarray]:
        import torch

        self._use_category(category)

        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"expected an HxWx3 RGB image, got shape {arr.shape}; grayscale is converted "
                "to 3-channel upstream (protocol §3)"
            )
        # Raw [0,1] at native resolution: the model's own pre-processor resizes to 240 (with
        # antialiasing, point 4) and CLIP-normalises inside forward. See the module docstring.
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
        tensor = tensor.unsqueeze(0).to(self._device)   # 1x3xHxW

        with torch.no_grad():
            out = self._model(tensor)   # InferenceBatch(pred_score, anomaly_map) at 240x240

        score = float(out.pred_score.reshape(-1)[0].item())
        amap = out.anomaly_map.detach().cpu().numpy().reshape(
            out.anomaly_map.shape[-2], out.anomaly_map.shape[-1]
        ).astype(np.float32)
        return score, amap
