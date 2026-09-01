import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domains.pcb.preprocessing.dataset import DeepPCBDataset


DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "raw" / "deeppcb"
MANIFEST_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "manifest.csv"
CLASS_MAP_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "class_map.json"


def test_dataset_initialization():
    """Test that dataset initializes correctly for each split."""
    for split in ['train', 'val', 'test']:
        dataset = DeepPCBDataset(
            data_root=str(DATA_ROOT),
            manifest_path=str(MANIFEST_PATH),
            split=split,
            class_map_path=str(CLASS_MAP_PATH)
        )
        assert len(dataset) > 0, f"{split} split should have samples"


def test_dataset_lengths():
    """Test that dataset lengths match expected split sizes."""
    train_ds = DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), 'train', str(CLASS_MAP_PATH))
    val_ds = DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), 'val', str(CLASS_MAP_PATH))
    test_ds = DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), 'test', str(CLASS_MAP_PATH))

    assert len(train_ds) == 1050, f"Train split should have 1050 samples, got {len(train_ds)}"
    assert len(val_ds) == 225, f"Val split should have 225 samples, got {len(val_ds)}"
    assert len(test_ds) == 225, f"Test split should have 225 samples, got {len(test_ds)}"


def test_invalid_split():
    """Test that invalid split names raise an error."""
    with pytest.raises(AssertionError):
        DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), 'invalid_split')


def test_sample_structure():
    """Test that samples have the correct structure."""
    dataset = DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), 'train', str(CLASS_MAP_PATH))
    sample = dataset[0]

    # Check keys
    assert 'defect_image' in sample
    assert 'template_image' in sample
    assert 'annotations' in sample
    assert 'pair_id' in sample
    assert 'group' in sample
    assert 'classes' in sample

    # Check types
    assert hasattr(sample['defect_image'], 'size'), "defect_image should be a PIL Image"
    assert hasattr(sample['template_image'], 'size'), "template_image should be a PIL Image"
    assert isinstance(sample['annotations'], list), "annotations should be a list"
    assert isinstance(sample['pair_id'], str), "pair_id should be a string"
    assert isinstance(sample['group'], str), "group should be a string"
    assert isinstance(sample['classes'], set), "classes should be a set"


def test_image_dimensions():
    """Test that loaded images have expected dimensions (640x640)."""
    dataset = DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), 'train', str(CLASS_MAP_PATH))
    sample = dataset[0]

    defect_img = sample['defect_image']
    template_img = sample['template_image']

    assert defect_img.size == (640, 640), f"Defect image should be 640x640, got {defect_img.size}"
    assert template_img.size == (640, 640), f"Template image should be 640x640, got {template_img.size}"


def test_annotation_structure():
    """Test that annotations have the correct structure."""
    dataset = DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), 'train', str(CLASS_MAP_PATH))
    sample = dataset[0]

    if len(sample['annotations']) > 0:
        anno = sample['annotations'][0]
        assert 'class_id' in anno
        assert 'x_center' in anno
        assert 'y_center' in anno
        assert 'width' in anno
        assert 'height' in anno

        # Check value ranges
        assert 0 <= anno['class_id'] <= 5, "class_id should be 0-5"
        assert 0 <= anno['x_center'] <= 1, "x_center should be normalized"
        assert 0 <= anno['y_center'] <= 1, "y_center should be normalized"
        assert 0 < anno['width'] <= 1, "width should be normalized and positive"
        assert 0 < anno['height'] <= 1, "height should be normalized and positive"


def test_class_mapping():
    """Test that class mapping works correctly."""
    dataset = DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), 'train', str(CLASS_MAP_PATH))

    expected_classes = {
        0: "open_circuit",
        1: "short_circuit",
        2: "mouse_bite",
        3: "spur",
        4: "spurious_copper",
        5: "pin_hole"
    }

    for class_id, class_name in expected_classes.items():
        assert dataset.get_class_name(class_id) == class_name


def test_index_out_of_range():
    """Test that out-of-range indices raise IndexError."""
    dataset = DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), 'train', str(CLASS_MAP_PATH))

    with pytest.raises(IndexError):
        _ = dataset[-1]

    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_train_val_test_isolation():
    """Test that train/val/test splits have no overlap."""
    train_ds = DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), 'train', str(CLASS_MAP_PATH))
    val_ds = DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), 'val', str(CLASS_MAP_PATH))
    test_ds = DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), 'test', str(CLASS_MAP_PATH))

    train_ids = {train_ds[i]['pair_id'] for i in range(len(train_ds))}
    val_ids = {val_ds[i]['pair_id'] for i in range(len(val_ds))}
    test_ids = {test_ds[i]['pair_id'] for i in range(len(test_ds))}

    assert len(train_ids & val_ids) == 0, "Train and val should not overlap"
    assert len(train_ids & test_ids) == 0, "Train and test should not overlap"
    assert len(val_ids & test_ids) == 0, "Val and test should not overlap"


def test_all_samples_accessible():
    """Test that all samples in each split can be accessed."""
    for split in ['train', 'val', 'test']:
        dataset = DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), split, str(CLASS_MAP_PATH))
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample is not None, f"Sample {i} in {split} should not be None"


def test_get_split_info():
    """Test that get_split_info returns correct information."""
    dataset = DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), 'train', str(CLASS_MAP_PATH))
    info = dataset.get_split_info()

    assert info['split'] == 'train'
    assert info['num_samples'] == 1050
    assert 'data_root' in info


def test_portable_path_handling():
    """Test that portable forward-slash paths are resolved correctly."""
    dataset = DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), 'train', str(CLASS_MAP_PATH))
    sample = dataset[0]

    # Should load successfully despite forward slashes in manifest
    assert sample['defect_image'] is not None
    assert sample['template_image'] is not None
    assert len(sample['annotations']) >= 0  # May be empty but should parse


def test_deterministic_access():
    """Test that repeated access to same index returns same data."""
    dataset = DeepPCBDataset(str(DATA_ROOT), str(MANIFEST_PATH), 'train', str(CLASS_MAP_PATH))

    sample1 = dataset[0]
    sample2 = dataset[0]

    assert sample1['pair_id'] == sample2['pair_id']
    assert sample1['group'] == sample2['group']
    assert sample1['classes'] == sample2['classes']
    assert len(sample1['annotations']) == len(sample2['annotations'])
