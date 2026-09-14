"""Tests for PCB domain configuration constants."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domains.pcb.config import (
    CLASS_NAME_TO_ID,
    CLASS_NAMES,
    DEFAULT_COLOR_JITTER,
    DEFAULT_HFLIP_P,
    DEFAULT_ROTATION_ANGLES,
    DEFAULT_VFLIP_P,
    EXPECTED_SPLIT_SIZES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_ANNOTATIONS_PER_IMAGE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_CHANNELS,
    NUM_CLASSES,
    SPLIT_RATIOS,
)


def test_image_dimensions():
    """Image dimensions match DeepPCB native resolution."""
    assert IMAGE_WIDTH == 640
    assert IMAGE_HEIGHT == 640
    assert NUM_CHANNELS == 3


def test_class_mapping_completeness():
    """CLASS_NAMES covers all 6 defect classes (0-5)."""
    assert NUM_CLASSES == 6
    assert len(CLASS_NAMES) == NUM_CLASSES
    for i in range(NUM_CLASSES):
        assert i in CLASS_NAMES, f"Missing class_id {i}"
        assert isinstance(CLASS_NAMES[i], str)


def test_inverse_mapping_consistent():
    """CLASS_NAME_TO_ID is a faithful inverse of CLASS_NAMES."""
    assert len(CLASS_NAME_TO_ID) == len(CLASS_NAMES)
    for cid, name in CLASS_NAMES.items():
        assert CLASS_NAME_TO_ID[name] == cid


def test_split_ratios_sum_to_one():
    """Train/val/test ratios sum to 1.0."""
    assert abs(sum(SPLIT_RATIOS.values()) - 1.0) < 1e-9


def test_expected_split_sizes():
    """Split sizes match 1500 total pairs."""
    total = sum(EXPECTED_SPLIT_SIZES.values())
    assert total == 1500


def test_normalization_shape():
    """Normalization vectors have 3 elements (one per RGB channel)."""
    assert len(NORMALIZE_MEAN) == 3
    assert len(NORMALIZE_STD) == 3
    for v in NORMALIZE_STD:
        assert v > 0, "Standard deviations must be positive"


def test_augmentation_defaults():
    """Augmentation defaults are reasonable."""
    assert 0 <= DEFAULT_HFLIP_P <= 1
    assert 0 <= DEFAULT_VFLIP_P <= 1
    assert 0 in DEFAULT_ROTATION_ANGLES, "Identity rotation must be included"
    for angle in DEFAULT_ROTATION_ANGLES:
        assert angle % 90 == 0, "Only 90-degree multiples are supported"
    for k in ("brightness", "contrast", "saturation", "hue"):
        assert k in DEFAULT_COLOR_JITTER
        assert DEFAULT_COLOR_JITTER[k] >= 0


def test_max_annotations_positive():
    """MAX_ANNOTATIONS_PER_IMAGE is a positive integer."""
    assert isinstance(MAX_ANNOTATIONS_PER_IMAGE, int)
    assert MAX_ANNOTATIONS_PER_IMAGE > 0
