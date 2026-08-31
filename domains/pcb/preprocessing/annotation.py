from typing import Tuple

def parse_deeppcb_annotation(line: str, img_width: int = 640, img_height: int = 640) -> Tuple[int, float, float, float, float]:
    """
    Parses a single line of a DeepPCB annotation file.
    Format: x1,y1,x2,y2,type

    Args:
        line: Raw comma-separated string from DeepPCB .txt file
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (class_id, x_center, y_center, width, height)
        Where coordinates are normalized to [0, 1].
    """
    parts = line.strip().split(',')
    if len(parts) != 5:
        raise ValueError(f"Expected 5 comma-separated values, got {len(parts)}: '{line}'")

    x1, y1, x2, y2, defect_type = map(int, parts)

    # Convert from 1-indexed to 0-indexed
    class_id = defect_type - 1

    # Calculate normalized YOLO format
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / float(img_width)
    height = (y2 - y1) / float(img_height)

    return class_id, x_center, y_center, width, height
