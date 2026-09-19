"""Deterministic training data path for the PCB anomaly foundation."""

import csv
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset
from PIL import Image

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


class SyntheticNormalTransform:
    """Create deterministic, mild non-defect variation from a template."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def __call__(self, template: torch.Tensor, index: int) -> torch.Tensor:
        generator = torch.Generator(device=template.device).manual_seed(self.seed + index)
        contrast = 0.96 + 0.08 * torch.rand((), generator=generator, device=template.device)
        brightness = -0.02 + 0.04 * torch.rand((), generator=generator, device=template.device)
        varied = (template - 0.5) * contrast + 0.5 + brightness
        noise = 0.005 * torch.randn(
            template.shape, generator=generator, dtype=template.dtype, device=template.device
        )
        return (varied + noise).clamp(0.0, 1.0)


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
            "annotations": sample["annotations"],
            "pair_id": sample["pair_id"],
            "group": sample["group"],
        }


def anomaly_collate_fn(batch):
    """Stack fixed-size anomaly inputs without detector annotations."""
    collated = {
        "pair_inputs": torch.stack([sample["pair_input"] for sample in batch]),
        "normal_targets": torch.stack([sample["normal_target"] for sample in batch]),
        "pair_ids": [sample["pair_id"] for sample in batch],
        "groups": [sample["group"] for sample in batch],
    }
    if "annotations" in batch[0]:
        collated["annotations"] = [sample["annotations"] for sample in batch]
    return collated


class DeepPCBTemplateAnomalyDataset(Dataset):
    """Template-only synthetic-normal samples for denoising autoencoder use.

    Only manifest rows and golden-template files are read by this training
    path.  Corresponding inspected images and annotations are not loaded.
    """

    ALLOWED_SPLITS = ("train", "val")

    def __init__(
        self,
        data_root: str,
        manifest_path: str,
        split: str,
        class_map_path: Optional[str] = None,
        pair_ids: Optional[Sequence[str]] = None,
        variation_seed: int = 42,
    ):
        if split not in self.ALLOWED_SPLITS:
            raise ValueError(
                "Template anomaly training accepts only train or val; the test split is isolated"
            )
        self.split = split
        self.data_root = Path(data_root)
        with open(manifest_path, newline="", encoding="utf-8") as manifest_file:
            self.samples = [
                row for row in csv.DictReader(manifest_file) if row["split"] == split
            ]
        if not self.samples:
            raise ValueError(f"No samples found for split '{split}' in {manifest_path}")
        available_ids = [sample["pair_id"] for sample in self.samples]
        if pair_ids is None:
            self.indices = list(range(len(available_ids)))
        else:
            requested = set(pair_ids)
            unknown = requested - set(available_ids)
            if unknown:
                raise ValueError(f"Unknown pair IDs for split '{split}': {sorted(unknown)[:3]}")
            self.indices = [index for index, pair_id in enumerate(available_ids) if pair_id in requested]
        self.pair_ids = [available_ids[index] for index in self.indices]
        self.synthetic_transform = SyntheticNormalTransform(variation_seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[self.indices[index]]
        template_image = Image.open(self.data_root / sample["temp_img"]).convert("RGB")
        _, template = PairedToTensor()(template_image, template_image)
        return {
            "pair_input": self.synthetic_transform(template, self.indices[index]),
            "normal_target": template,
            "pair_id": sample["pair_id"],
            "group": sample["group"],
        }
