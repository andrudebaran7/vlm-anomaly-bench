"""anomalib PatchCore backend for PatchCoreRef (GPU/Colab only).

Bridges the repo's injectable seam - fit(train_images) / score(image) -> (float, HxW map) - to
anomalib, which is directory-and-datamodule oriented. `fit` materialises the streamed normal
images to a temp directory and runs anomalib's PatchCore training to fill the coreset memory bank;
`score` hands one native-resolution image to the model as a raw [0,1] tensor and lets the model's
own PreProcessor resize and normalise it - the same single pre-processing `fit` gets - then returns
anomalib's RAW pred_score and anomaly_map (protocol v0.2.6 - no per-image normalisation;
PatchCoreRef upsamples the map to native resolution).

anomalib and torch are imported lazily so this module imports in CI (it is never constructed
there). Hyperparameters come from configs/methods/patchcore_ref.yaml.
"""
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

#: The backbone's effective stride over the layer2/layer3 features PatchCore concatenates.
#: MEASURED, not assumed: the 2026-09-16 reproduction run printed 102.4 coreset points per
#: training image at `coreset_sampling_ratio: 0.1`, i.e. 1024 patches from a 256x256 input,
#: i.e. a 32x32 grid -- 256 / 32 = 8.
_FEATURE_STRIDE = 8

#: The two pre-processings this backend can run. `anomalib` is what anomalib 2.6.0 does on its
#: own (`Resize([256, 256]) + Normalize`, recorded in the 2026-08-21 Colab session) and is the
#: default because it is what every number measured so far came from. `classic` is what
#: PatchCore's paper describes and what the 2026-09-16 gate failure pre-registered as suspect.
PREPROCESS_CHOICES = ("anomalib", "classic")


def preprocess_spec(name: str) -> dict:
    """What a named pre-processing does, and the patch count that follows from it.

    Separate from the backend class, and dependency-free, for two reasons: it is the only part
    of this module CPU can test, and the patch count is what a Colab cell checks against BEFORE
    paying for a 40-minute fit. A run that silently used the other transform would still print
    a plausible number -- that is precisely how the double-normalisation bug survived for five
    days -- so the check has to be on something measurable, and this is where that number is
    defined once.
    """
    if name not in PREPROCESS_CHOICES:
        raise ValueError(
            f"unknown preprocess {name!r}; choices are {list(PREPROCESS_CHOICES)}. This is "
            "checked before anomalib is imported so a typo in a notebook cell fails at once "
            "rather than after a fit."
        )
    # Classic PatchCore (arXiv:2106.08265 §4.1): resize the short side to 256, then centre-crop
    # to 224. anomalib 2.6.0 resizes the whole frame to 256x256 and never crops.
    center_crop = 224 if name == "classic" else None
    side = center_crop or 256
    return {
        "name": name,
        "resize": 256,
        "center_crop": center_crop,
        "patches_per_image": (side // _FEATURE_STRIDE) ** 2,
    }


class PatchCoreBackend:
    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: tuple[str, ...] = ("layer2", "layer3"),
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        device: str = "cuda",
        num_workers: int = 2,
        seed: int | None = None,
        preprocess: str = "anomalib",
    ):
        # BEFORE the lazy import, so a typo fails in a millisecond rather than after the
        # install, the weights and a 40-minute fit.
        spec = preprocess_spec(preprocess)

        from anomalib.models import Patchcore  # lazy: GPU-only
        from anomalib.engine import Engine

        self._device = device
        self._num_workers = num_workers
        # PatchCore's coreset sampling is stochastic: two fits on byte-identical inputs give
        # different memory banks and different scores (probe stage 11, 2026-08-26: the same
        # image scored 202.796432 and then 203.890701 across two runs). Protocol §6 requires
        # three seeds wherever stochasticity exists, so it has to be controllable from here.
        #
        # The default is None -- NOT a silent 0. run_eval.py already records a `seed` in every
        # shard's provenance without applying it to anything, and a backend that quietly seeded
        # itself would make that record look true while the two numbers still had nothing to do
        # with each other. None means "unseeded, and the provenance is the caller's problem";
        # an int means the run is reproducible and says so.
        # Public: PatchCoreRef reads this to declare, in the shard's provenance, the seed that
        # was actually applied. A private name would force the adapter to reach through it.
        self.seed = seed
        self._model = Patchcore(
            backbone=backbone,
            layers=list(layers),
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
        )
        # The Engine configuration below is EXACTLY the one verified to work on
        # anomalib 2.6.0 / Colab T4 (2026-08-21), reached by testing four candidates and
        # measuring which filled the coreset. Nothing is added on top of it: three separate
        # failures this session came from passing kwargs that looked reasonable and were
        # never run.
        #
        #   * No `task=`. Engine.__init__ takes **kwargs and forwards them to Lightning's
        #     Trainer, built lazily inside engine.train(), so task= constructs fine and dies
        #     one call later with TypeError: Trainer.__init__() got an unexpected keyword
        #     argument 'task'. A swallowed kwarg is worse than a rejected one -- the
        #     traceback points at fit(), not at the constructor that accepted it.
        #   * No `enable_checkpointing=False`. anomalib installs its own ModelCheckpoint and
        #     Lightning refuses the combination outright.
        #   * `enable_progress_bar=False` is REQUIRED, and is the subtle one. Lightning's
        #     RichProgressBar holds a live display while anomalib's coreset sampler writes
        #     its own tqdm into it; in a notebook the two nest until
        #     RecursionError: maximum recursion depth exceeded, raised from rich's renderer
        #     while the bar reprints pinned at 0/N. Raising the recursion limit does NOT
        #     help (tested at 20000) -- the nesting is unbounded, not deep-but-finite.
        #   * `limit_val_batches=0` + `num_sanity_val_steps=0` because the datamodule is
        #     built with ValSplitMode.NONE and so never creates `val_data`, while
        #     Lightning's fit loop sets up the validation loop unconditionally.
        #   * No accelerator/devices/max_epochs: not in the verified configuration. anomalib
        #     picks the GPU on its own and stops at max_epochs=1 for PatchCore.
        self._engine = Engine(
            logger=False,
            enable_progress_bar=False,
            limit_val_batches=0,
            num_sanity_val_steps=0,
        )
        # No pre-processing transform is kept here on purpose. AnomalibModule.forward is
        #     batch = self.pre_processor(batch) if self.pre_processor else batch
        # so the model pre-processes whatever it is handed. See score().
        #
        # Public, like `seed`, and for the same reason: PatchCoreRef reads it to declare in
        # every shard's provenance which pre-processing actually ran.
        self.preprocess = spec["name"]
        if spec["center_crop"] is not None:
            self._apply_classic_preprocessing(spec)
        self._fitted = False

    def _apply_classic_preprocessing(self, spec: dict) -> None:
        """Replace the model's own transform with classic PatchCore's Resize -> CenterCrop.

        It goes through the PreProcessor anomalib itself built on the model, and reuses that
        object's own Normalize, rather than importing a PreProcessor class by a path taken
        from a documentation snippet. anomalib 2.6.0's Engine API was verified on 2026-08-21;
        its PreProcessor API was NOT, and this backend has already been bitten five separate
        times by a call that looked reasonable and had never been run. Probe stage 13 prints
        the real API before any fit is paid for.

        Both halves of the model see this: `pre_processor` is a child module of the
        LightningModule, so `fit` gets it through anomalib's own pipeline and `score` gets it
        through `forward`. A transform applied to only one of them would build the memory
        bank at one resolution and query it at another.

        **Consequence for pixel metrics, recorded rather than discovered later:** a centre
        crop means the anomaly map covers only the middle 224/256 of the frame, and
        PatchCoreRef then upsamples it across the WHOLE native image. The reproduction gate
        is image-level I-AUROC and is unaffected; any pixel-level number from a `classic`
        run would be spatially wrong, and this is why the choice is recorded per shard.
        """
        from torchvision.transforms.v2 import CenterCrop, Compose, Resize

        pre = getattr(self._model, "pre_processor", None)
        transform = getattr(pre, "transform", None)
        if transform is None:
            raise RuntimeError(
                f"preprocess='classic' cannot be applied: the model's pre_processor is "
                f"{pre!r}, which exposes no `transform` to replace. Run probe stage 13 "
                "(notebooks/probe_patchcore.py) to print the real API on the installed "
                "anomalib, and fix this from what it prints -- not from the docs."
            )
        # Carry over anomalib's OWN Normalize instead of re-stating ImageNet's constants:
        # those constants are not in this repo's verified record, and a second copy of them
        # is a second thing that can drift from what the untouched path applies.
        steps = list(getattr(transform, "transforms", [transform]))
        normalize = [t for t in steps if type(t).__name__ == "Normalize"]
        if not normalize:
            raise RuntimeError(
                f"preprocess='classic' cannot be applied: the model's transform is {steps}, "
                "which contains no Normalize to carry over. Dropping normalisation silently "
                "would produce a complete run of plausible, wrong numbers."
            )
        # Resize with a single int takes the SHORT side to 256 and preserves aspect, which
        # is what the paper describes; Resize([256, 256]) -- anomalib's own -- does not.
        pre.transform = Compose([
            Resize(spec["resize"], antialias=True),
            CenterCrop(spec["center_crop"]),
            *normalize,
        ])

    def fit(self, train_images: Iterable[np.ndarray]) -> None:
        import torch
        from anomalib.data import Folder
        from anomalib.data.utils import TestSplitMode, ValSplitMode
        from PIL import Image

        # Materialise the streamed normal images to a temp dir anomalib's Folder can read.
        self._tmp = tempfile.TemporaryDirectory()
        good = Path(self._tmp.name) / "good"
        good.mkdir(parents=True)
        n = 0
        for i, img in enumerate(train_images):
            Image.fromarray(np.asarray(img, dtype=np.uint8)).save(good / f"{i:05d}.png")
            n += 1
        if n == 0:
            raise ValueError("PatchCoreBackend.fit received no training images")

        # No `task=` here: anomalib 2.x removed it from Folder. Its own docs are
        # inconsistent - the API reference has no such parameter while several
        # snippet pages still pass task="classification" - and the snippets are what
        # this backend was first written from. Verified 2026-08-21 in a Colab session:
        # passing it raises TypeError: Folder.__init__() got an unexpected keyword
        # argument 'task'.
        datamodule = Folder(
            name="fit",
            root=self._tmp.name,
            normal_dir="good",
            # anomalib defaults to 8; Colab has 2 CPUs and the DataLoader warns twice per
            # setup about it. 2 is what the verified run used.
            num_workers=self._num_workers,
            val_split_mode=ValSplitMode.NONE,
            test_split_mode=TestSplitMode.NONE,
        )
        if self.seed is not None:
            # workers=True also seeds the DataLoader worker processes, which matters because
            # the coreset sampler draws inside them.
            from lightning import seed_everything
            seed_everything(self.seed, workers=True)

        datamodule.setup()
        self._engine.train(model=self._model, datamodule=datamodule)

        # Lightning owns device placement during fit and hands the model back on CPU when it
        # finishes, but score() calls the model directly with a CUDA tensor:
        #   RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
        #   (torch.FloatTensor) should be the same
        # So reclaim it here. Verified 2026-08-21.
        self._model.to(self._device)
        self._model.eval()

        # The coreset lives on the inner model as `memory_bank`. If anomalib registered it as
        # a buffer, .to() above moved it; if it is a plain tensor attribute, it did not, and
        # the nearest-neighbour search would fail the same way one call later. Checked rather
        # than assumed, because a half-moved model is the kind of thing that fails deep.
        inner = getattr(self._model, "model", None)
        bank = getattr(inner, "memory_bank", None)
        if isinstance(bank, torch.Tensor) and bank.numel():
            want = torch.device(self._device)
            if bank.device.type != want.type:
                inner.memory_bank = bank.to(want)

        self._fitted = True

    def score(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        import torch

        if not self._fitted:
            raise RuntimeError("PatchCoreBackend.score called before fit")

        # Hand the model a RAW image and let it pre-process. `PreProcessor` is a child module
        # of the Patchcore LightningModule and AnomalibModule.forward runs it unconditionally:
        #     batch = self.pre_processor(batch) if self.pre_processor else batch
        #     batch = self.model(batch)
        #     return self.post_processor(batch) if self.post_processor else batch
        # so anything we pre-process here is resized and ImageNet-normalised a SECOND time.
        #
        # This was the shipped behaviour until 2026-08-26 and it produced no error, no warning
        # and a perfectly plausible score: the injected-defect smoke test separated correctly
        # either way, because normalising twice is monotonic enough to keep the ordering. What
        # it would have broken is the VisA reproduction gate, with nothing pointing back here.
        # Measured in probe stage 11: model(pre_processed) equals model.model(normalised twice)
        # to the last digit, while model.model(pre_processed) equals model(raw).
        #
        # Only the conversion HWC uint8 -> CHW float in [0,1] is ours; the pre-processor's
        # Resize([256,256]) does the rest, exactly as it does for every image during fit. The
        # image goes in at native resolution for that reason.
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"expected an HxWx3 RGB image, got shape {arr.shape}; grayscale is converted "
                "to 3-channel upstream (protocol §3)"
            )
        # torch warns here that `arr` is not writable (PIL buffers are read-only) and that
        # writing through the tensor would be undefined. Nothing does: `.float()` converts
        # uint8 -> float32, and a float32 tensor cannot alias a uint8 buffer (4 bytes per
        # element against 1), so it is obliged to allocate. `.div_` then mutates that fresh
        # copy, never the image. Do NOT silence the warning with `arr.copy()` -- that buys
        # nothing and adds a second full copy of a 1400x1900x3 array per scored image.
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
        tensor = tensor.unsqueeze(0).to(self._device)   # 1x3xHxW, raw [0,1]

        with torch.no_grad():
            out = self._model(tensor)   # -> InferenceBatch(pred_score, anomaly_map, ...)

        score = float(out.pred_score.reshape(-1)[0].item())
        amap = out.anomaly_map.detach().cpu().numpy().reshape(
            out.anomaly_map.shape[-2], out.anomaly_map.shape[-1]
        ).astype(np.float32)
        return score, amap
