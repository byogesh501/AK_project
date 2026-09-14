"""
PCB preprocessing pipeline — concrete implementation of BasePreprocessor.

Wires together the dataset loader, paired transforms, and DataLoader with
custom collation into a single entry point.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from core.preprocessing.base import BasePreprocessor
from domains.pcb.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)
from domains.pcb.preprocessing.collate import pcb_collate_fn
from domains.pcb.preprocessing.dataset import DeepPCBDataset
from domains.pcb.preprocessing.transforms import (
    TrainTransform,
    ValTestTransform,
)


class PCBPreprocessor(BasePreprocessor):
    """
    Domain-specific preprocessor for PCB / DeepPCB images.

    Implements the ``BasePreprocessor`` contract and additionally
    provides :meth:`create_dataloader` for end-to-end pipeline usage.

    Args:
        data_root: Path to ``data/raw/deeppcb/``
        manifest_path: Path to ``data/processed/manifest.csv``
        class_map_path: Optional path to ``data/processed/class_map.json``
        normalize: Whether to apply channel normalization.
    """

    def __init__(
        self,
        data_root: str,
        manifest_path: str,
        class_map_path: Optional[str] = None,
        normalize: bool = True,
    ):
        self.data_root = Path(data_root)
        self.manifest_path = manifest_path
        self.class_map_path = class_map_path
        self.normalize = normalize

        self._train_transform = TrainTransform(normalize=normalize)
        self._val_transform = ValTestTransform(normalize=normalize)

    # ------------------------------------------------------------------
    # BasePreprocessor interface
    # ------------------------------------------------------------------

    def load_image(self, path: str) -> Image.Image:
        """Load a single PCB image as an RGB PIL Image."""
        return Image.open(path).convert("RGB")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Apply deterministic val/test preprocessing to a single image.

        Returns a numpy array of shape ``(C, H, W)`` in ``[0, 1]``
        (or normalized) float32.
        """
        from domains.pcb.preprocessing.transforms import PairedToTensor, PairedNormalize

        to_tensor = PairedToTensor()
        tensor, _ = to_tensor(image, image)  # pair with itself (ignored)
        if self.normalize:
            norm = PairedNormalize()
            tensor, _ = norm(tensor, tensor)
        return tensor.numpy()

    def augment(self, image: Image.Image) -> List[np.ndarray]:
        """
        Return one augmented copy (as numpy) of the image.

        **Note:** This method satisfies the ``BasePreprocessor`` interface but
        does NOT perform paired augmentation. For training with synchronized
        defect/template augmentation, use :meth:`create_dataloader` which
        applies ``TrainTransform`` to both images together.
        """
        arr = self.preprocess(image)
        return [arr]

    def get_config(self) -> Dict[str, Any]:
        """Return the current preprocessing configuration."""
        return {
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "normalize": self.normalize,
            "normalize_mean": NORMALIZE_MEAN,
            "normalize_std": NORMALIZE_STD,
            "data_root": str(self.data_root),
        }

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    def create_dataset(self, split: str) -> DeepPCBDataset:
        """
        Create a ``DeepPCBDataset`` for the given split with the
        appropriate transform already attached.
        """
        transform = self._train_transform if split == "train" else self._val_transform
        return DeepPCBDataset(
            data_root=str(self.data_root),
            manifest_path=self.manifest_path,
            split=split,
            class_map_path=self.class_map_path,
            transform=transform,
        )

    def create_dataloader(
        self,
        split: str,
        batch_size: int = 8,
        shuffle: Optional[bool] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        """
        Build a ready-to-iterate ``DataLoader`` for *split*.

        Shuffling defaults to ``True`` for train and ``False`` otherwise.
        Uses :func:`pcb_collate_fn` so variable-length annotations are
        padded and masked automatically.
        """
        dataset = self.create_dataset(split)
        if shuffle is None:
            shuffle = split == "train"
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=pcb_collate_fn,
        )
