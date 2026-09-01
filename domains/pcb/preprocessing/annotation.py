from typing import Tuple
import re

def parse_deeppcb_annotation(line: str, img_width: int = 640, img_height: int = 640) -> Tuple[int, float, float, float, float]:
    """
    Parses a single line of a DeepPCB annotation file.
    Format is documented as comma-separated but exists in the wild as space-separated.
    We handle both via regex split on whitespace or commas.

    Args:
        line: Raw string from DeepPCB .txt file
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (class_id, x_center, y_center, width, height)
        Where coordinates are normalized to [0, 1].
    """
    # Split by any whitespace or comma
    parts = [p for p in re.split(r'[\s,]+', line.strip()) if p]
    if len(parts) != 5:
        raise ValueError(f"Expected 5 values, got {len(parts)}: '{line}'")

    x1, y1, x2, y2, defect_type = map(int, parts)

    if x2 <= x1:
        raise ValueError(f"Invalid x coordinates: x2 ({x2}) must be > x1 ({x1})")
    if y2 <= y1:
        raise ValueError(f"Invalid y coordinates: y2 ({y2}) must be > y1 ({y1})")

    if not (1 <= defect_type <= 6):
        raise ValueError(f"Invalid defect_type {defect_type}. Must be between 1 and 6.")

    # Convert from 1-indexed to 0-indexed
    class_id = defect_type - 1

    # Calculate normalized YOLO format
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / float(img_width)
    height = (y2 - y1) / float(img_height)

    return class_id, x_center, y_center, width, height
