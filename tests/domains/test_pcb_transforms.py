import pytest
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domains.pcb.preprocessing.transforms import (
    PairedToTensor,
    PairedNormalize,
    PairedRandomHorizontalFlip,
    PairedRandomVerticalFlip,
    PairedRandomRotation,
    PairedColorJitter,
    TrainTransform,
    ValTestTransform
)


def create_test_image(size=(640, 640), color=(255, 0, 0)):
    """Create a test PIL image."""
    return Image.new('RGB', size, color)


def create_test_annotations():
    """Create test annotations in normalized YOLO format."""
    return [
        {'class_id': 0, 'x_center': 0.5, 'y_center': 0.5, 'width': 0.2, 'height': 0.2},
        {'class_id': 1, 'x_center': 0.3, 'y_center': 0.7, 'width': 0.1, 'height': 0.15}
    ]


def test_paired_to_tensor():
    """Test PairedToTensor conversion."""
    transform = PairedToTensor()
    defect_img = create_test_image()
    template_img = create_test_image(color=(0, 255, 0))

    defect_tensor, template_tensor = transform(defect_img, template_img)

    # Check type and shape
    assert isinstance(defect_tensor, torch.Tensor)
    assert isinstance(template_tensor, torch.Tensor)
    assert defect_tensor.shape == (3, 640, 640)
    assert template_tensor.shape == (3, 640, 640)

    # Check value range [0, 1]
    assert 0 <= defect_tensor.min() <= 1
    assert 0 <= defect_tensor.max() <= 1


def test_paired_normalize():
    """Test PairedNormalize."""
    transform = PairedNormalize()
    defect_tensor = torch.rand(3, 640, 640)
    template_tensor = torch.rand(3, 640, 640)

    defect_norm, template_norm = transform(defect_tensor, template_tensor)

    # Check shape unchanged
    assert defect_norm.shape == (3, 640, 640)
    assert template_norm.shape == (3, 640, 640)

    # Check normalization applied (values should be centered around 0)
    assert defect_norm.mean().abs() < 1.0  # Roughly centered


def test_horizontal_flip_bbox_transform():
    """Test that horizontal flip correctly transforms bounding boxes."""
    transform = PairedRandomHorizontalFlip(p=1.0)  # Always flip
    defect_img = create_test_image()
    template_img = create_test_image()
    annotations = create_test_annotations()

    _, _, flipped_annos = transform(defect_img, template_img, annotations)

    # Check that x_center is flipped: new_x = 1 - old_x
    for orig, flipped in zip(annotations, flipped_annos):
        assert abs(flipped['x_center'] - (1.0 - orig['x_center'])) < 1e-6
        assert flipped['y_center'] == orig['y_center']  # y unchanged
        assert flipped['width'] == orig['width']  # dimensions unchanged
        assert flipped['height'] == orig['height']
        assert flipped['class_id'] == orig['class_id']  # class unchanged


def test_vertical_flip_bbox_transform():
    """Test that vertical flip correctly transforms bounding boxes."""
    transform = PairedRandomVerticalFlip(p=1.0)  # Always flip
    defect_img = create_test_image()
    template_img = create_test_image()
    annotations = create_test_annotations()

    _, _, flipped_annos = transform(defect_img, template_img, annotations)

    # Check that y_center is flipped: new_y = 1 - old_y
    for orig, flipped in zip(annotations, flipped_annos):
        assert flipped['x_center'] == orig['x_center']  # x unchanged
        assert abs(flipped['y_center'] - (1.0 - orig['y_center'])) < 1e-6
        assert flipped['width'] == orig['width']
        assert flipped['height'] == orig['height']
        assert flipped['class_id'] == orig['class_id']


def test_rotation_90_bbox_transform():
    """Test that 90-degree rotation correctly transforms bounding boxes."""
    transform = PairedRandomRotation(angles=[90])  # Always 90°
    defect_img = create_test_image()
    template_img = create_test_image()
    annotations = [{'class_id': 0, 'x_center': 0.3, 'y_center': 0.7, 'width': 0.2, 'height': 0.1}]

    _, _, rotated_annos = transform(defect_img, template_img, annotations)

    # 90° rotation: (x, y) -> (y, 1-x), swap width/height
    orig = annotations[0]
    rotated = rotated_annos[0]

    assert abs(rotated['x_center'] - orig['y_center']) < 1e-6
    assert abs(rotated['y_center'] - (1.0 - orig['x_center'])) < 1e-6
    assert abs(rotated['width'] - orig['height']) < 1e-6
    assert abs(rotated['height'] - orig['width']) < 1e-6
    assert rotated['class_id'] == orig['class_id']


def test_rotation_180_bbox_transform():
    """Test that 180-degree rotation correctly transforms bounding boxes."""
    transform = PairedRandomRotation(angles=[180])  # Always 180°
    defect_img = create_test_image()
    template_img = create_test_image()
    annotations = [{'class_id': 0, 'x_center': 0.3, 'y_center': 0.7, 'width': 0.2, 'height': 0.1}]

    _, _, rotated_annos = transform(defect_img, template_img, annotations)

    # 180° rotation: (x, y) -> (1-x, 1-y), dimensions unchanged
    orig = annotations[0]
    rotated = rotated_annos[0]

    assert abs(rotated['x_center'] - (1.0 - orig['x_center'])) < 1e-6
    assert abs(rotated['y_center'] - (1.0 - orig['y_center'])) < 1e-6
    assert rotated['width'] == orig['width']
    assert rotated['height'] == orig['height']
    assert rotated['class_id'] == orig['class_id']


def test_paired_image_synchronization():
    """Test that spatial transforms keep defect and template synchronized."""
    transform = PairedRandomHorizontalFlip(p=1.0)

    # Create distinguishable images
    defect_img = create_test_image(color=(255, 0, 0))
    template_img = create_test_image(color=(0, 255, 0))
    annotations = create_test_annotations()

    defect_flipped, template_flipped, _ = transform(defect_img, template_img, annotations)

    # Both should be flipped
    assert defect_flipped.size == defect_img.size
    assert template_flipped.size == template_img.size


def test_color_jitter_preserves_annotations():
    """Test that color jitter doesn't affect annotations."""
    transform = PairedColorJitter()
    defect_img = create_test_image()
    template_img = create_test_image()

    defect_jittered, template_jittered = transform(defect_img, template_img)

    # Images should still be PIL Images with same size
    assert isinstance(defect_jittered, Image.Image)
    assert isinstance(template_jittered, Image.Image)
    assert defect_jittered.size == defect_img.size
    assert template_jittered.size == template_img.size


def test_train_transform_output_types():
    """Test that TrainTransform produces correct output types."""
    transform = TrainTransform()
    defect_img = create_test_image()
    template_img = create_test_image()
    annotations = create_test_annotations()

    defect_tensor, template_tensor, aug_annos = transform(defect_img, template_img, annotations)

    # Check outputs are tensors
    assert isinstance(defect_tensor, torch.Tensor)
    assert isinstance(template_tensor, torch.Tensor)
    assert isinstance(aug_annos, list)

    # Check shapes
    assert defect_tensor.shape == (3, 640, 640)
    assert template_tensor.shape == (3, 640, 640)

    # Check annotations preserved
    assert len(aug_annos) == len(annotations)


def test_val_test_transform_deterministic():
    """Test that ValTestTransform is deterministic."""
    transform = ValTestTransform()
    defect_img = create_test_image()
    template_img = create_test_image()
    annotations = create_test_annotations()

    # Run transform twice
    result1 = transform(defect_img, template_img, annotations)
    result2 = transform(defect_img, template_img, annotations)

    # Should be identical
    assert torch.allclose(result1[0], result2[0])  # defect tensor
    assert torch.allclose(result1[1], result2[1])  # template tensor
    assert result1[2] == result2[2]  # annotations unchanged


def test_val_test_transform_no_augmentation():
    """Test that ValTestTransform doesn't augment annotations."""
    transform = ValTestTransform()
    defect_img = create_test_image()
    template_img = create_test_image()
    annotations = create_test_annotations()

    _, _, output_annos = transform(defect_img, template_img, annotations)

    # Annotations should be exactly the same
    assert output_annos == annotations


def test_train_vs_val_behavior():
    """Test that train uses augmentation while val/test doesn't."""
    train_transform = TrainTransform(hflip_p=1.0, vflip_p=0.0, rotation_angles=[0])
    val_transform = ValTestTransform()

    defect_img = create_test_image()
    template_img = create_test_image()
    annotations = [{'class_id': 0, 'x_center': 0.3, 'y_center': 0.5, 'width': 0.2, 'height': 0.1}]

    # Train should flip
    _, _, train_annos = train_transform(defect_img, template_img, annotations)
    assert train_annos[0]['x_center'] != annotations[0]['x_center']  # Should be flipped

    # Val should not change
    _, _, val_annos = val_transform(defect_img, template_img, annotations)
    assert val_annos == annotations


def test_annotation_value_ranges():
    """Test that transformed annotations maintain valid ranges."""
    transform = TrainTransform()
    defect_img = create_test_image()
    template_img = create_test_image()
    annotations = create_test_annotations()

    _, _, aug_annos = transform(defect_img, template_img, annotations)

    for anno in aug_annos:
        # Check normalized coordinates are in [0, 1]
        assert 0 <= anno['x_center'] <= 1
        assert 0 <= anno['y_center'] <= 1
        assert 0 < anno['width'] <= 1
        assert 0 < anno['height'] <= 1
        # Check class_id preserved
        assert anno['class_id'] in [0, 1, 2, 3, 4, 5]


def test_output_tensor_ranges():
    """Test that output tensors have expected value ranges."""
    # Without normalization
    transform_no_norm = ValTestTransform(normalize=False)
    defect_img = create_test_image()
    template_img = create_test_image()
    annotations = []

    defect_tensor, template_tensor, _ = transform_no_norm(defect_img, template_img, annotations)

    # Should be in [0, 1] range
    assert 0 <= defect_tensor.min() <= 1
    assert 0 <= defect_tensor.max() <= 1

    # With normalization
    transform_norm = ValTestTransform(normalize=True)
    defect_tensor_norm, _, _ = transform_norm(defect_img, template_img, annotations)

    # Normalized values can be negative
    assert defect_tensor_norm.mean().abs() < 10  # Reasonable range after normalization
