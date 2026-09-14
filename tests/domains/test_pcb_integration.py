"""
Integration tests: dataset + transforms + collate + DataLoader.

These tests use the real DeepPCB data on disk and verify the full pipeline
from raw images to batched tensors.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domains.pcb.preprocessing.collate import pcb_collate_fn
from domains.pcb.preprocessing.dataset import DeepPCBDataset
from domains.pcb.preprocessing.preprocessor import PCBPreprocessor
from domains.pcb.preprocessing.transforms import TrainTransform, ValTestTransform
from domains.pcb.config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES

DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "raw" / "deeppcb"
MANIFEST_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "manifest.csv"
CLASS_MAP_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "class_map.json"

_data_available = DATA_ROOT.exists() and MANIFEST_PATH.exists()
skip_no_data = pytest.mark.skipif(not _data_available, reason="DeepPCB data not available")


# ---------------------------------------------------------------------------
# Dataset + Transform integration
# ---------------------------------------------------------------------------

@skip_no_data
class TestDatasetTransformIntegration:
    """Verify that the dataset passes paired images through transforms."""

    def test_train_sample_returns_tensors(self):
        """Train split with TrainTransform returns tensors, not PIL Images."""
        transform = TrainTransform(normalize=True)
        ds = DeepPCBDataset(
            str(DATA_ROOT), str(MANIFEST_PATH), "train",
            str(CLASS_MAP_PATH), transform=transform,
        )
        sample = ds[0]
        assert isinstance(sample["defect_image"], torch.Tensor)
        assert isinstance(sample["template_image"], torch.Tensor)
        assert sample["defect_image"].shape == (3, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert sample["template_image"].shape == (3, IMAGE_HEIGHT, IMAGE_WIDTH)

    def test_val_sample_returns_tensors(self):
        """Val split with ValTestTransform returns tensors."""
        transform = ValTestTransform(normalize=True)
        ds = DeepPCBDataset(
            str(DATA_ROOT), str(MANIFEST_PATH), "val",
            str(CLASS_MAP_PATH), transform=transform,
        )
        sample = ds[0]
        assert isinstance(sample["defect_image"], torch.Tensor)
        assert sample["defect_image"].shape == (3, IMAGE_HEIGHT, IMAGE_WIDTH)

    def test_annotations_survive_transform(self):
        """Annotations are still valid dicts after going through transform."""
        transform = TrainTransform(normalize=True)
        ds = DeepPCBDataset(
            str(DATA_ROOT), str(MANIFEST_PATH), "train",
            str(CLASS_MAP_PATH), transform=transform,
        )
        sample = ds[0]
        for anno in sample["annotations"]:
            assert "class_id" in anno
            assert 0 <= anno["class_id"] < NUM_CLASSES
            assert 0 <= anno["x_center"] <= 1
            assert 0 <= anno["y_center"] <= 1
            assert 0 < anno["width"] <= 1
            assert 0 < anno["height"] <= 1

    def test_val_annotations_unchanged(self):
        """ValTestTransform does not modify annotation values."""
        ds_raw = DeepPCBDataset(
            str(DATA_ROOT), str(MANIFEST_PATH), "val",
            str(CLASS_MAP_PATH),
        )
        ds_transformed = DeepPCBDataset(
            str(DATA_ROOT), str(MANIFEST_PATH), "val",
            str(CLASS_MAP_PATH), transform=ValTestTransform(normalize=True),
        )
        raw_annos = ds_raw[0]["annotations"]
        xf_annos = ds_transformed[0]["annotations"]
        assert len(raw_annos) == len(xf_annos)
        for r, t in zip(raw_annos, xf_annos):
            assert r == t


# ---------------------------------------------------------------------------
# DataLoader + collate integration
# ---------------------------------------------------------------------------

@skip_no_data
class TestDataLoaderIntegration:
    """Full pipeline: dataset → transform → DataLoader → collated batch."""

    def test_train_dataloader_batch(self):
        """One train batch has correct shapes and types."""
        transform = TrainTransform(normalize=True)
        ds = DeepPCBDataset(
            str(DATA_ROOT), str(MANIFEST_PATH), "train",
            str(CLASS_MAP_PATH), transform=transform,
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, shuffle=False, collate_fn=pcb_collate_fn,
        )
        batch = next(iter(loader))

        assert batch["defect_images"].shape == (4, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert batch["template_images"].shape == (4, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert batch["annotations"].dim() == 3  # [B, max_annos, 5]
        assert batch["annotations"].shape[0] == 4
        assert batch["annotations"].shape[2] == 5
        assert batch["anno_mask"].shape[:2] == batch["annotations"].shape[:2]
        assert len(batch["pair_ids"]) == 4

    def test_val_dataloader_batch(self):
        """Val batch is deterministic and correctly shaped."""
        transform = ValTestTransform(normalize=True)
        ds = DeepPCBDataset(
            str(DATA_ROOT), str(MANIFEST_PATH), "val",
            str(CLASS_MAP_PATH), transform=transform,
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=2, shuffle=False, collate_fn=pcb_collate_fn,
        )
        b1 = next(iter(loader))
        b2 = next(iter(loader))

        # Re-iterate and first batch should be identical (deterministic)
        loader2 = torch.utils.data.DataLoader(
            ds, batch_size=2, shuffle=False, collate_fn=pcb_collate_fn,
        )
        b1_again = next(iter(loader2))
        assert torch.allclose(b1["defect_images"], b1_again["defect_images"])
        assert torch.allclose(b1["template_images"], b1_again["template_images"])


# ---------------------------------------------------------------------------
# PCBPreprocessor pipeline tests
# ---------------------------------------------------------------------------

@skip_no_data
class TestPCBPreprocessorPipeline:
    """End-to-end tests using PCBPreprocessor as the entry point."""

    def _preprocessor(self):
        return PCBPreprocessor(
            data_root=str(DATA_ROOT),
            manifest_path=str(MANIFEST_PATH),
            class_map_path=str(CLASS_MAP_PATH),
            normalize=True,
        )

    def test_create_dataset_train(self):
        """create_dataset('train') returns a usable dataset."""
        pp = self._preprocessor()
        ds = pp.create_dataset("train")
        assert len(ds) == 1050
        sample = ds[0]
        assert isinstance(sample["defect_image"], torch.Tensor)

    def test_create_dataset_val(self):
        """create_dataset('val') returns a usable dataset."""
        pp = self._preprocessor()
        ds = pp.create_dataset("val")
        assert len(ds) == 225

    def test_create_dataloader_iterates(self):
        """create_dataloader produces at least one batch."""
        pp = self._preprocessor()
        loader = pp.create_dataloader("train", batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert "defect_images" in batch
        assert batch["defect_images"].shape[0] == 4

    def test_create_dataloader_val_no_shuffle(self):
        """Val dataloader defaults to no shuffling."""
        pp = self._preprocessor()
        loader = pp.create_dataloader("val", batch_size=2)
        # Just confirm it's iterable and deterministic
        b1 = next(iter(loader))
        b2 = next(iter(pp.create_dataloader("val", batch_size=2)))
        assert torch.allclose(b1["defect_images"], b2["defect_images"])

    def test_full_epoch_no_crash(self):
        """Iterate over 10 val samples without errors."""
        pp = self._preprocessor()
        ds = pp.create_dataset("val")
        for i in range(min(10, len(ds))):
            sample = ds[i]
            assert sample["defect_image"].shape == (3, IMAGE_HEIGHT, IMAGE_WIDTH)
