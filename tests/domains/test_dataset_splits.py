import pytest
import json
import csv
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.create_splits import (
    find_all_pairs,
    stratified_split,
    compute_stats,
    validate_splits,
    DATA_ROOT,
    OUT_DIR,
    SEED,
    CLASS_NAMES
)


def test_split_determinism():
    """Test that splits are deterministic with the same seed."""
    pairs = find_all_pairs(DATA_ROOT)

    # Generate splits twice with same seed
    splits1, pair_classes1 = stratified_split(pairs, SEED)
    splits2, pair_classes2 = stratified_split(pairs, SEED)

    # Extract pair IDs for comparison
    for split_name in ["train", "val", "test"]:
        ids1 = {f"{p['group']}_{p['base_name']}" for p in splits1[split_name]}
        ids2 = {f"{p['group']}_{p['base_name']}" for p in splits2[split_name]}
        assert ids1 == ids2, f"Split {split_name} is not deterministic"


def test_split_counts():
    """Test that splits have correct counts (1050/225/225 for 1500 pairs)."""
    pairs = find_all_pairs(DATA_ROOT)
    assert len(pairs) == 1500, f"Expected 1500 pairs, got {len(pairs)}"

    splits, _ = stratified_split(pairs, SEED)

    # Check approximate 70/15/15 split
    assert len(splits["train"]) >= 1040 and len(splits["train"]) <= 1060
    assert len(splits["val"]) >= 220 and len(splits["val"]) <= 230
    assert len(splits["test"]) >= 220 and len(splits["test"]) <= 235

    # Check total
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == 1500, f"Total pairs {total} != 1500"


def test_no_pair_leakage():
    """Test that no pair appears in multiple splits."""
    pairs = find_all_pairs(DATA_ROOT)
    splits, _ = stratified_split(pairs, SEED)

    train_ids = {f"{p['group']}_{p['base_name']}" for p in splits["train"]}
    val_ids = {f"{p['group']}_{p['base_name']}" for p in splits["val"]}
    test_ids = {f"{p['group']}_{p['base_name']}" for p in splits["test"]}

    # Check no overlap
    assert len(train_ids & val_ids) == 0, "Train/val overlap detected"
    assert len(train_ids & test_ids) == 0, "Train/test overlap detected"
    assert len(val_ids & test_ids) == 0, "Val/test overlap detected"

    # Check all pairs accounted for
    all_split_ids = train_ids | val_ids | test_ids
    all_pair_ids = {f"{p['group']}_{p['base_name']}" for p in pairs}
    assert all_split_ids == all_pair_ids, "Not all pairs accounted for in splits"


def test_all_classes_represented():
    """Test that all 6 classes appear in every split."""
    pairs = find_all_pairs(DATA_ROOT)
    splits, pair_classes = stratified_split(pairs, SEED)

    for split_name in ["train", "val", "test"]:
        split_ids = {f"{p['group']}_{p['base_name']}" for p in splits[split_name]}
        classes_in_split = set()
        for pair_id in split_ids:
            classes_in_split.update(pair_classes.get(pair_id, set()))

        assert len(classes_in_split) == 6, f"Split {split_name} missing classes: {set(CLASS_NAMES.keys()) - classes_in_split}"


def test_manifest_consistency():
    """Test that manifest.csv matches splits.json."""
    if not (OUT_DIR / "manifest.csv").exists() or not (OUT_DIR / "splits.json").exists():
        pytest.skip("Output files not generated yet")

    # Load splits.json
    with open(OUT_DIR / "splits.json", 'r') as f:
        splits_json = json.load(f)

    # Load manifest.csv
    manifest_splits = {"train": set(), "val": set(), "test": set()}
    with open(OUT_DIR / "manifest.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest_splits[row["split"]].add(row["pair_id"])

    # Compare
    for split_name in ["train", "val", "test"]:
        json_ids = set(splits_json[split_name])
        csv_ids = manifest_splits[split_name]
        assert json_ids == csv_ids, f"Mismatch in {split_name} between splits.json and manifest.csv"


def test_class_map_completeness():
    """Test that class_map.json contains all 6 classes."""
    if not (OUT_DIR / "class_map.json").exists():
        pytest.skip("class_map.json not generated yet")

    with open(OUT_DIR / "class_map.json", 'r') as f:
        class_map = json.load(f)

    assert len(class_map) == 6, f"Expected 6 classes, got {len(class_map)}"

    expected_classes = {
        "0": "open_circuit",
        "1": "short_circuit",
        "2": "mouse_bite",
        "3": "spur",
        "4": "spurious_copper",
        "5": "pin_hole"
    }

    assert class_map == expected_classes, "class_map.json does not match expected mapping"


def test_stats_json_structure():
    """Test that stats.json has correct structure and reasonable values."""
    if not (OUT_DIR / "stats.json").exists():
        pytest.skip("stats.json not generated yet")

    with open(OUT_DIR / "stats.json", 'r') as f:
        stats = json.load(f)

    # Check structure
    assert "train" in stats and "val" in stats and "test" in stats

    for split_name in ["train", "val", "test"]:
        assert "num_pairs" in stats[split_name]
        assert "class_distribution" in stats[split_name]

        # Check all classes present
        dist = stats[split_name]["class_distribution"]
        for class_name in CLASS_NAMES.values():
            assert class_name in dist, f"Class {class_name} missing from {split_name} stats"
            assert dist[class_name] > 0, f"Class {class_name} has zero count in {split_name}"


def test_validation_passes():
    """Test that the validation function reports no errors."""
    pairs = find_all_pairs(DATA_ROOT)
    splits, pair_classes = stratified_split(pairs, SEED)

    errors = validate_splits(splits, pairs, pair_classes)
    assert len(errors) == 0, f"Validation errors: {errors}"
