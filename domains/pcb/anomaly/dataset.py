"""Deterministic training data path for the PCB anomaly foundation."""

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from domains.pcb.anomaly.model import build_pair_difference_input
from domains.pcb.preprocessing.dataset import DeepPCBDataset
from domains.pcb.preprocessing.transforms import PairedToTensor


class AnomalyPairTransform:
    """Convert an aligned pair to unnormalized tensors without augmentation.

    Pixel-level differencing needs the original shared RGB scale.  This avoids
    the detector's random color jitter and ImageNet-style normalization.
    """

    def __init__(self):
        self.to_tensor = PairedToTensor()

    def __call__(self, defect_image, template_image, annotations):
        defect_tensor, template_tensor = self.to_tensor(defect_image, template_image)
        return defect_tensor, template_tensor, annotations


class DeepPCBAnomalyDataset(Dataset):
    """DeepPCB pairs with golden templates as normal reconstruction targets.

    DeepPCB's manifest has no annotation-free pairs.  Each selected pair is
    therefore represented by its defect/template/difference input while the
    target is only the paired golden template.  Defective pixels are never
    targets.  This initial training path intentionally accepts train and
    validation only so the preserved test split cannot enter training or model
    selection.
    """

    ALLOWED_SPLITS = ("train", "val")

    def __init__(
        self,
        data_root: str,
        manifest_path: str,
        split: str,
        class_map_path: Optional[str] = None,
    ):
        if split not in self.ALLOWED_SPLITS:
            raise ValueError(
                "Anomaly foundation accepts only train or val; the test split is isolated"
            )
        self.split = split
        self.data_root = Path(data_root)
        self.dataset = DeepPCBDataset(
            data_root=data_root,
            manifest_path=manifest_path,
            split=split,
            class_map_path=class_map_path,
            transform=AnomalyPairTransform(),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.dataset[index]
        defect_image = sample["defect_image"]
        template_image = sample["template_image"]
        return {
            "pair_input": build_pair_difference_input(defect_image, template_image),
            "normal_target": template_image,
            "pair_id": sample["pair_id"],
            "group": sample["group"],
        }


def anomaly_collate_fn(batch):
    """Stack fixed-size anomaly inputs without detector annotations."""
    return {
        "pair_inputs": torch.stack([sample["pair_input"] for sample in batch]),
        "normal_targets": torch.stack([sample["normal_target"] for sample in batch]),
        "pair_ids": [sample["pair_id"] for sample in batch],
        "groups": [sample["group"] for sample in batch],
    }
