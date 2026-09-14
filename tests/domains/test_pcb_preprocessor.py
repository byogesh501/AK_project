"""Tests for pcb_collate_fn and PCBPreprocessor."""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domains.pcb.preprocessing.collate import pcb_collate_fn
from domains.pcb.preprocessing.preprocessor import PCBPreprocessor
from domains.pcb.config import IMAGE_WIDTH, IMAGE_HEIGHT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(pair_id="p1", num_annos=2):
    """Build a single sample dict that looks like DeepPCBDataset output
    after a transform has been applied (tensors, not PIL images)."""
    defect = torch.rand(3, IMAGE_HEIGHT, IMAGE_WIDTH)
    template = torch.rand(3, IMAGE_HEIGHT, IMAGE_WIDTH)
    annotations = [
        {
            "class_id": i % 6,
            "x_center": 0.1 * (i + 1),
            "y_center": 0.2 * (i + 1),
            "width": 0.05,
            "height": 0.05,
        }
        for i in range(num_annos)
    ]
    return {
        "defect_image": defect,
        "template_image": template,
        "annotations": annotations,
        "pair_id": pair_id,
        "group": "groupX",
        "classes": {"open_circuit"},
    }


# ---------------------------------------------------------------------------
# pcb_collate_fn tests
# ---------------------------------------------------------------------------

class TestPCBCollateFn:
    """Tests for the custom collate function."""

    def test_batch_shapes(self):
        """Stacked image tensors have shape [B, C, H, W]."""
        batch = [_make_sample(f"p{i}", num_annos=3) for i in range(4)]
        out = pcb_collate_fn(batch)
        assert out["defect_images"].shape == (4, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert out["template_images"].shape == (4, 3, IMAGE_HEIGHT, IMAGE_WIDTH)

    def test_annotation_padding(self):
        """Annotations are padded to the longest sequence in the batch."""
        batch = [
            _make_sample("p0", num_annos=1),
            _make_sample("p1", num_annos=5),
            _make_sample("p2", num_annos=3),
        ]
        out = pcb_collate_fn(batch)
        assert out["annotations"].shape == (3, 5, 5)  # B=3, max_annos=5, feats=5

    def test_annotation_mask(self):
        """anno_mask marks real annotations True, padding False."""
        batch = [
            _make_sample("p0", num_annos=1),
            _make_sample("p1", num_annos=3),
        ]
        out = pcb_collate_fn(batch)
        mask = out["anno_mask"]
        # First sample: 1 real, rest padded
        assert mask[0, 0].item() is True
        assert mask[0, 1].item() is False
        # Second sample: 3 real
        assert mask[1, :3].all()

    def test_empty_annotations(self):
        """Handles samples with zero annotations (e.g. defect-free)."""
        sample = _make_sample("p0", num_annos=0)
        batch = [sample]
        out = pcb_collate_fn(batch)
        # Should still produce a valid tensor (min 1 slot)
        assert out["annotations"].shape[0] == 1
        assert out["annotations"].shape[1] >= 1
        assert out["anno_mask"].sum().item() == 0

    def test_metadata_passthrough(self):
        """pair_ids, groups, and classes are plain lists."""
        batch = [_make_sample(f"p{i}") for i in range(2)]
        out = pcb_collate_fn(batch)
        assert out["pair_ids"] == ["p0", "p1"]
        assert len(out["groups"]) == 2
        assert all(isinstance(c, set) for c in out["classes"])

    def test_annotation_values_preserved(self):
        """Real annotation values survive collation."""
        sample = _make_sample("p0", num_annos=1)
        anno = sample["annotations"][0]
        out = pcb_collate_fn([sample])
        row = out["annotations"][0, 0]
        assert row[0].item() == anno["class_id"]
        assert abs(row[1].item() - anno["x_center"]) < 1e-6
        assert abs(row[2].item() - anno["y_center"]) < 1e-6
        assert abs(row[3].item() - anno["width"]) < 1e-6
        assert abs(row[4].item() - anno["height"]) < 1e-6


# ---------------------------------------------------------------------------
# PCBPreprocessor unit tests (no real data needed)
# ---------------------------------------------------------------------------

class TestPCBPreprocessor:
    """Unit tests for PCBPreprocessor (no filesystem access)."""

    def _dummy_preprocessor(self, tmp_path):
        """Create a preprocessor with dummy paths (won't load data)."""
        return PCBPreprocessor(
            data_root=str(tmp_path),
            manifest_path=str(tmp_path / "manifest.csv"),
            normalize=True,
        )

    def test_inherits_base_preprocessor(self):
        """PCBPreprocessor is a valid BasePreprocessor subclass."""
        from core.preprocessing.base import BasePreprocessor
        assert issubclass(PCBPreprocessor, BasePreprocessor)

    def test_load_image(self, tmp_path):
        """load_image opens an RGB PIL Image."""
        # Create a tiny test image on disk
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (32, 32), (100, 150, 200)).save(img_path)

        pp = self._dummy_preprocessor(tmp_path)
        img = pp.load_image(str(img_path))
        assert img.mode == "RGB"
        assert img.size == (32, 32)

    def test_preprocess_output_shape(self, tmp_path):
        """preprocess returns a (C, H, W) float numpy array."""
        pp = self._dummy_preprocessor(tmp_path)
        img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (128, 128, 128))
        arr = pp.preprocess(img)
        assert arr.shape == (3, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert arr.dtype.kind == "f"  # float32

    def test_augment_returns_list(self, tmp_path):
        """
        augment returns a list of numpy arrays.

        Note: augment() satisfies BasePreprocessor interface but does NOT
        perform paired augmentation. Use create_dataloader() for training.
        """
        pp = self._dummy_preprocessor(tmp_path)
        img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT))
        result = pp.augment(img)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert result[0].shape == (3, IMAGE_HEIGHT, IMAGE_WIDTH)

    def test_get_config(self, tmp_path):
        """get_config returns expected keys."""
        pp = self._dummy_preprocessor(tmp_path)
        cfg = pp.get_config()
        assert cfg["image_width"] == IMAGE_WIDTH
        assert cfg["image_height"] == IMAGE_HEIGHT
        assert cfg["normalize"] is True
        assert "normalize_mean" in cfg
        assert "normalize_std" in cfg
