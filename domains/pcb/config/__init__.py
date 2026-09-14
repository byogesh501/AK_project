"""
PCB Configuration

PCB-specific settings: defect categories, board types,
inspection parameters, and domain constants.
"""

# Image dimensions (DeepPCB native resolution)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
NUM_CHANNELS = 3

# Defect class mapping (0-indexed class_id -> human-readable name)
NUM_CLASSES = 6
CLASS_NAMES = {
    0: "open_circuit",
    1: "short_circuit",
    2: "mouse_bite",
    3: "spur",
    4: "spurious_copper",
    5: "pin_hole",
}

# Inverse mapping: name -> class_id
CLASS_NAME_TO_ID = {name: cid for cid, name in CLASS_NAMES.items()}

# Dataset split sizes (deterministic seed=42)
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
EXPECTED_SPLIT_SIZES = {"train": 1050, "val": 225, "test": 225}

# Normalization parameters (ImageNet-scale — reasonable for RGB PCB images
# that will be trained from scratch; the model will learn to compensate)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Training augmentation defaults
DEFAULT_HFLIP_P = 0.5
DEFAULT_VFLIP_P = 0.5
DEFAULT_ROTATION_ANGLES = [0, 90, 180, 270]
DEFAULT_COLOR_JITTER = {
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.1,
    "hue": 0.05,
}

# Maximum number of annotations per image (for padding in collate)
MAX_ANNOTATIONS_PER_IMAGE = 50
