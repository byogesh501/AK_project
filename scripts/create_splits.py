#!/usr/bin/env python3
"""
Create deterministic 70/15/15 splits for DeepPCB dataset.
Generates: splits.json, class_map.json, stats.json, manifest.csv
"""
import os
import json
import csv
import random
from collections import defaultdict
from pathlib import Path

# Import our existing parser
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from domains.pcb.preprocessing.annotation import parse_deeppcb_annotation

DATA_ROOT = Path(__file__).parent.parent / "data" / "raw" / "deeppcb"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

CLASS_NAMES = {
    0: "open_circuit",
    1: "short_circuit",
    2: "mouse_bite",
    3: "spur",
    4: "spurious_copper",
    5: "pin_hole"
}

def find_all_pairs(data_root):
    """Find all valid (test_img, temp_img, anno) triplets."""
    pairs = []
    group_dirs = [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("group")]

    for group_dir in group_dirs:
        group_id = group_dir.name.replace("group", "")
        img_dir = group_dir / group_id
        anno_dir = group_dir / f"{group_id}_not"

        if not img_dir.exists() or not anno_dir.exists():
            continue

        test_imgs = list(img_dir.glob("*_test.jpg"))

        for test_img in test_imgs:
            base = test_img.stem.replace("_test", "")
            temp_img = img_dir / f"{base}_temp.jpg"
            anno_file = anno_dir / f"{base}.txt"

            if temp_img.exists() and anno_file.exists():
                pairs.append({
                    "base_name": base,
                    "group": group_dir.name,
                    "test_img": str(test_img.relative_to(data_root)),
                    "temp_img": str(temp_img.relative_to(data_root)),
                    "anno_file": str(anno_file.relative_to(data_root))
                })

    return pairs

def get_classes_in_image(anno_path):
    """Parse annotation file and return set of class IDs present."""
    classes = set()
    full_path = DATA_ROOT / anno_path
    try:
        with open(full_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                class_id, *_ = parse_deeppcb_annotation(line)
                classes.add(class_id)
    except Exception as e:
        print(f"Warning: Failed to parse {anno_path}: {e}")
    return classes

def stratified_split(pairs, seed=42):
    """
    Perform deterministic stratified split ensuring class balance across splits.

    Algorithm:
    1. Build stable pair identity using group + base_name
    2. Parse classes for each pair (fail loudly on parse errors)
    3. Group pairs by their primary class (first class ID when sorted)
    4. Within each class group, deterministically assign to train/val/test
       maintaining 70/15/15 ratios per class
    5. This ensures every class is proportionally represented in all splits
    """
    random.seed(seed)

    # Get classes per pair with error handling
    pair_classes = {}
    parse_failures = []
    for p in pairs:
        pair_id = f"{p['group']}_{p['base_name']}"
        classes = get_classes_in_image(p["anno_file"])
        if not classes:
            parse_failures.append(p["anno_file"])
        pair_classes[pair_id] = classes

    if parse_failures:
        raise ValueError(f"Failed to parse classes from {len(parse_failures)} annotation files: {parse_failures[:5]}")

    # Group pairs by their primary class (lowest class ID for determinism)
    class_to_pairs = defaultdict(list)
    for p in pairs:
        pair_id = f"{p['group']}_{p['base_name']}"
        classes = pair_classes[pair_id]
        primary_class = min(classes)  # Use lowest class ID as primary
        class_to_pairs[primary_class].append(p)

    # Shuffle each class group independently with the same seed for determinism
    for class_id in class_to_pairs:
        pairs_in_class = class_to_pairs[class_id]
        # Sort first for determinism across platforms, then shuffle
        pairs_in_class.sort(key=lambda x: (x['group'], x['base_name']))
        random.Random(seed + class_id).shuffle(pairs_in_class)

    # Distribute each class group across splits maintaining 70/15/15 ratio
    train_pairs = []
    val_pairs = []
    test_pairs = []

    for class_id in sorted(class_to_pairs.keys()):
        class_pairs = class_to_pairs[class_id]
        n = len(class_pairs)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        train_pairs.extend(class_pairs[:n_train])
        val_pairs.extend(class_pairs[n_train:n_train + n_val])
        test_pairs.extend(class_pairs[n_train + n_val:])

    return {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs
    }, pair_classes

def compute_stats(splits, pair_classes):
    """Compute class distribution and counts per split."""
    stats = {}
    for split_name, split_pairs in splits.items():
        class_dist = defaultdict(int)
        for p in split_pairs:
            pair_id = f"{p['group']}_{p['base_name']}"
            classes = pair_classes.get(pair_id, set())
            for c in classes:
                class_dist[CLASS_NAMES[c]] += 1

        stats[split_name] = {
            "num_pairs": len(split_pairs),
            "class_distribution": dict(class_dist)
        }
    return stats

def write_outputs(splits, pair_classes, stats):
    """Write all output files."""

    # 1. splits.json
    splits_json = {k: [f"{p['group']}_{p['base_name']}" for p in v] for k, v in splits.items()}
    with open(OUT_DIR / "splits.json", 'w') as f:
        json.dump(splits_json, f, indent=2)

    # 2. class_map.json
    class_map = {str(k): v for k, v in CLASS_NAMES.items()}
    with open(OUT_DIR / "class_map.json", 'w') as f:
        json.dump(class_map, f, indent=2)

    # 3. stats.json
    with open(OUT_DIR / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # 4. manifest.csv
    with open(OUT_DIR / "manifest.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["split", "pair_id", "group", "test_img", "temp_img", "anno_file", "classes"])
        for split_name, split_pairs in splits.items():
            for p in split_pairs:
                pair_id = f"{p['group']}_{p['base_name']}"
                classes = pair_classes.get(pair_id, set())
                class_str = ";".join(CLASS_NAMES[c] for c in sorted(classes))
                writer.writerow([
                    split_name, pair_id, p["group"],
                    p["test_img"], p["temp_img"], p["anno_file"],
                    class_str
                ])

def validate_splits(splits, pairs, pair_classes):
    """Validate splits for correctness."""
    all_ids = set()
    errors = []

    for split_name, split_pairs in splits.items():
        ids = {f"{p['group']}_{p['base_name']}" for p in split_pairs}
        # Check duplicates within split
        if len(ids) != len(split_pairs):
            errors.append(f"Duplicate pairs in {split_name}")
        # Check overlap with other splits
        overlap = all_ids & ids
        if overlap:
            errors.append(f"Pairs in {split_name} also in other splits: {overlap}")
        all_ids |= ids

    # Check all pairs accounted for
    pair_ids = {f"{p['group']}_{p['base_name']}" for p in pairs}
    missing = pair_ids - all_ids
    if missing:
        errors.append(f"Missing {len(missing)} pairs from splits: {list(missing)[:5]}")

    extra = all_ids - pair_ids
    if extra:
        errors.append(f"Extra pairs in splits: {extra}")

    # Check file existence
    for split_name, split_pairs in splits.items():
        for p in split_pairs:
            for key in ["test_img", "temp_img", "anno_file"]:
                path = DATA_ROOT / p[key]
                if not path.exists():
                    errors.append(f"Missing file: {path}")
                    break  # Don't spam errors for same pair

    # Check all classes are represented in each split
    for split_name in splits.keys():
        split_ids = {f"{p['group']}_{p['base_name']}" for p in splits[split_name]}
        classes_in_split = set()
        for pair_id in split_ids:
            classes_in_split.update(pair_classes.get(pair_id, set()))

        missing_classes = set(CLASS_NAMES.keys()) - classes_in_split
        if missing_classes:
            errors.append(f"Split '{split_name}' missing classes: {[CLASS_NAMES[c] for c in missing_classes]}")

    return errors

def main():
    print("Finding all valid pairs...")
    pairs = find_all_pairs(DATA_ROOT)
    print(f"Found {len(pairs)} valid pairs")

    print("Computing classes per pair...")
    pair_classes = {}
    for p in pairs:
        pair_classes[f"{p['group']}_{p['base_name']}"] = get_classes_in_image(p["anno_file"])

    print("Creating stratified splits...")
    splits, pair_classes = stratified_split(pairs, SEED)

    print("Computing statistics...")
    stats = compute_stats(splits, pair_classes)

    print("Validating splits...")
    errors = validate_splits(splits, pairs, pair_classes)
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("Validation passed!")

    print("Writing output files...")
    write_outputs(splits, pair_classes, stats)

    # Print summary
    print("\n=== SPLIT SUMMARY ===")
    print(f"Total pairs: {len(pairs)}")
    for split_name, split_pairs in splits.items():
        dist = stats[split_name]["class_distribution"]
        print(f"\n{split_name}: {len(split_pairs)} pairs ({len(split_pairs)/len(pairs)*100:.1f}%)")
        for cls, count in sorted(dist.items()):
            print(f"  {cls}: {count}")

    print("\n=== STRATIFICATION METHOD ===")
    print("Algorithm: Primary-class stratification")
    print("- Each pair is assigned a primary class (lowest class ID)")
    print("- Pairs within each class group are shuffled deterministically")
    print("- Each class group is split 70/15/15 independently")
    print("- This ensures proportional representation of all classes across splits")

    return 0

if __name__ == "__main__":
    exit(main())