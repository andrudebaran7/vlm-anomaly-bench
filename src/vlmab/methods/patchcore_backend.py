"""anomalib PatchCore backend for PatchCoreRef (GPU/Colab only).

Bridges the repo's injectable seam - fit(train_images) / score(image) -> (float, HxW map) - to
anomalib, which is directory-and-datamodule oriented. `fit` materialises the streamed normal
images to a temp directory and runs anomalib's PatchCore training to fill the coreset memory bank;
`score` pre-processes one image with PatchCore's own transform, runs the trained torch model
forward, and returns anomalib's RAW pred_score and anomaly_map (protocol v0.2.6 - no per-image
normalisation; PatchCoreRef upsamples the map to native resolution).

anomalib and torch are imported lazily so this module imports in CI (it is never constructed
there). Hyperparameters come from configs/methods/patchcore_ref.yaml.
"""
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


class PatchCoreBackend:
    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: tuple[str, ...] = ("layer2", "layer3"),
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        device: str = "cuda",
        num_workers: int = 2,
    ):
        from anomalib.models import Patchcore  # lazy: GPU-only
        from anomalib.engine import Engine

        self._device = device
        self._num_workers = num_workers
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
        self._transform = type(self._model).configure_pre_processor().transform
        self._fitted = False

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

        # The pre-processor is a torchvision v2 pipeline that carries NO ToTensor step --
        # on anomalib 2.6.0 it is Resize([256,256]) + Normalize(imagenet), and Normalize
        # rejects PIL images outright ("Normalize() does not support PIL images").
        # So the conversion happens here: HWC uint8 -> CHW float in [0,1], which is the
        # range the ImageNet mean/std in that Normalize are defined against.
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"expected an HxWx3 RGB image, got shape {arr.shape}; grayscale is converted "
                "to 3-channel upstream (protocol §3)"
            )
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
        tensor = self._transform(tensor).unsqueeze(0).to(self._device)  # 1x3xHxW

        with torch.no_grad():
            out = self._model(tensor)   # -> InferenceBatch(pred_score, anomaly_map, ...)

        score = float(out.pred_score.reshape(-1)[0].item())
        amap = out.anomaly_map.detach().cpu().numpy().reshape(
            out.anomaly_map.shape[-2], out.anomaly_map.shape[-1]
        ).astype(np.float32)
        return score, amap
