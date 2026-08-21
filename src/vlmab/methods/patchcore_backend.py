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
    ):
        from anomalib.models import Patchcore  # lazy: GPU-only
        from anomalib.engine import Engine

        self._device = device
        self._model = Patchcore(
            backbone=backbone,
            layers=list(layers),
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
        )
        # PatchCore trains for a single epoch to fill the memory bank; disable checkpoints/logging.
        # No `task=` here either. Engine.__init__ takes **kwargs and forwards them to
        # Lightning's Trainer, which is built lazily inside engine.train() -- so passing
        # task= constructs fine and then dies one call later with
        #   TypeError: Trainer.__init__() got an unexpected keyword argument 'task'
        # A swallowed kwarg is worse than a rejected one: the traceback points at
        # fit(), not at the constructor that accepted it. Verified 2026-08-21.
        self._engine = Engine(
            accelerator="gpu",
            devices=1,
            max_epochs=1,
            logger=False,
            # No enable_checkpointing=False: anomalib 2.x installs its own
            # ModelCheckpoint callback, and Lightning refuses the combination with
            #   MisconfigurationException: Trainer was configured with
            #   `enable_checkpointing=False` but found `ModelCheckpoint` in callbacks list.
            # Letting it write a checkpoint costs a little disk and nothing else.
            # Verified 2026-08-21.
        )
        self._transform = type(self._model).configure_pre_processor().transform
        self._fitted = False

    def fit(self, train_images: Iterable[np.ndarray]) -> None:
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
            val_split_mode=ValSplitMode.NONE,
            test_split_mode=TestSplitMode.NONE,
        )
        datamodule.setup()
        self._engine.train(model=self._model, datamodule=datamodule)
        self._model.eval()
        self._fitted = True

    def score(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        import torch
        from PIL import Image

        if not self._fitted:
            raise RuntimeError("PatchCoreBackend.score called before fit")

        tensor = self._transform(Image.fromarray(np.asarray(image, dtype=np.uint8)))
        tensor = tensor.unsqueeze(0).to(self._device)  # 1x3xHxW
        with torch.no_grad():
            out = self._model(tensor)   # -> InferenceBatch(pred_score, anomaly_map, ...)

        score = float(out.pred_score.reshape(-1)[0].item())
        amap = out.anomaly_map.detach().cpu().numpy().reshape(
            out.anomaly_map.shape[-2], out.anomaly_map.shape[-1]
        ).astype(np.float32)
        return score, amap
