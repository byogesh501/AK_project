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
    Perform deterministic stratified split with exact 1050/225/225 counts.

    Algorithm:
    1. Build stable pair identity using group + base_name
    2. Parse classes for each pair (fail loudly on parse errors)
    3. Sort all pairs deterministically by (group, base_name)
    4. Use iterative stratified splitting that considers full multi-label class sets
    5. Assign pairs sequentially to splits while balancing class distributions
    6. This ensures all classes are proportionally represented across splits
    """
    random.seed(seed)

    # Target exact counts
    TARGET_COUNTS = {"train": 1050, "val": 225, "test": 225}

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

    # Sort pairs deterministically, then shuffle with seed
    sorted_pairs = sorted(pairs, key=lambda x: (x['group'], x['base_name']))
    random.shuffle(sorted_pairs)

    # Count class occurrences per pair across all data
    total_class_counts = defaultdict(int)
    for classes in pair_classes.values():
        for c in classes:
            total_class_counts[c] += 1

    # Initialize splits and their class counts
    splits = {"train": [], "val": [], "test": []}
    split_class_counts = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int)
    }

    # Iteratively assign pairs to splits balancing class distributions
    for p in sorted_pairs:
        pair_id = f"{p['group']}_{p['base_name']}"
        classes = pair_classes[pair_id]

        # Find which split needs this pair most (considering class balance)
        best_split = None
        best_score = float('-inf')

        for split_name in ["train", "val", "test"]:
            # Skip if split is full
            if len(splits[split_name]) >= TARGET_COUNTS[split_name]:
                continue

            # Calculate how much this pair would improve class balance
            # Score = sum of (target_ratio - current_ratio) for each class in pair
            score = 0
            for c in classes:
                target_ratio = TARGET_COUNTS[split_name] / len(pairs)
                current_count = split_class_counts[split_name][c]
                current_ratio = current_count / total_class_counts[c] if total_class_counts[c] > 0 else 0
                score += (target_ratio - current_ratio)

            if best_split is None or score > best_score:
                best_split = split_name
                best_score = score

        # Assign to best split
        if best_split:
            splits[best_split].append(p)
            for c in classes:
                split_class_counts[best_split][c] += 1

    # Verify exact counts
    for split_name, target_count in TARGET_COUNTS.items():
        actual_count = len(splits[split_name])
        if actual_count != target_count:
            raise ValueError(f"Split {split_name} has {actual_count} pairs, expected {target_count}")

    return splits, pair_classes

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
    print("Algorithm: Greedy iterative multi-label stratification")
    print("- Pairs are shuffled deterministically with seed=42")
    print("- Each pair is assigned to the split that most needs its classes")
    print("- Assignment considers full multi-label class sets, not just primary class")
    print("- Scoring function balances class ratios across train/val/test")
    print("- Guarantees exact 1050/225/225 split with proportional class representation")

    return 0

if __name__ == "__main__":
    exit(main())