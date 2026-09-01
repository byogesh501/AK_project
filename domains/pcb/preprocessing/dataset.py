"""
PyTorch dataset loader for DeepPCB defect detection.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from domains.pcb.preprocessing.annotation import parse_deeppcb_annotation


class DeepPCBDataset(Dataset):
    """
    PyTorch dataset for DeepPCB template-paired defect detection.

    Each sample contains:
    - defect_image: The test PCB image with defects
    - template_image: The golden template (defect-free reference)
    - annotations: List of bounding boxes in normalized YOLO format
    - metadata: Additional info (pair_id, group, classes)

    Args:
        data_root: Path to data/raw/deeppcb/
        manifest_path: Path to data/processed/manifest.csv
        split: One of 'train', 'val', or 'test'
        class_map_path: Optional path to class_map.json
        transform: Optional transform to apply to images
    """

    def __init__(
        self,
        data_root: str,
        manifest_path: str,
        split: str,
        class_map_path: Optional[str] = None,
        transform: Optional[object] = None
    ):
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}"

        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        # Load class mapping
        if class_map_path:
            with open(class_map_path, 'r') as f:
                self.class_map = json.load(f)
        else:
            self.class_map = {
                "0": "open_circuit",
                "1": "short_circuit",
                "2": "mouse_bite",
                "3": "spur",
                "4": "spurious_copper",
                "5": "pin_hole"
            }

        # Load manifest and filter by split
        self.samples = self._load_manifest(manifest_path, split)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split '{split}' in {manifest_path}")

    def _load_manifest(self, manifest_path: str, split: str) -> List[Dict]:
        """Load and filter manifest by split."""
        samples = []
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    samples.append(row)
        return samples

    def _load_annotations(self, anno_path: Path) -> List[Dict]:
        """
        Load and parse annotations from file.

        Returns:
            List of dicts with keys: class_id, x_center, y_center, width, height
        """
        annotations = []
        with open(anno_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    class_id, x_c, y_c, w, h = parse_deeppcb_annotation(line)
                    annotations.append({
                        'class_id': class_id,
                        'x_center': x_c,
                        'y_center': y_c,
                        'width': w,
                        'height': h
                    })
                except Exception as e:
                    raise ValueError(f"Failed to parse line {line_num} in {anno_path}: {e}")

        return annotations

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get one sample.

        Returns:
            Dict with keys:
                - defect_image: PIL Image or transformed tensor
                - template_image: PIL Image or transformed tensor
                - annotations: List of annotation dicts
                - pair_id: Unique pair identifier
                - group: Group name
                - classes: Set of class names present
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")

        sample_info = self.samples[idx]

        # Resolve paths (manifest uses portable forward slashes)
        defect_path = self.data_root / sample_info['test_img']
        template_path = self.data_root / sample_info['temp_img']
        anno_path = self.data_root / sample_info['anno_file']

        # Load images
        try:
            defect_image = Image.open(defect_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Failed to load defect image {defect_path}: {e}")

        try:
            template_image = Image.open(template_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Failed to load template image {template_path}: {e}")

        # Load annotations
        try:
            annotations = self._load_annotations(anno_path)
        except Exception as e:
            raise ValueError(f"Failed to load annotations from {anno_path}: {e}")

        # Apply transforms if provided
        if self.transform:
            defect_image = self.transform(defect_image)
            template_image = self.transform(template_image)

        # Parse class names
        class_names = set(sample_info['classes'].split(';')) if sample_info['classes'] else set()

        return {
            'defect_image': defect_image,
            'template_image': template_image,
            'annotations': annotations,
            'pair_id': sample_info['pair_id'],
            'group': sample_info['group'],
            'classes': class_names
        }

    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        return self.class_map.get(str(class_id), f"unknown_{class_id}")

    def get_split_info(self) -> Dict:
        """Get information about this split."""
        return {
            'split': self.split,
            'num_samples': len(self.samples),
            'data_root': str(self.data_root)
        }
