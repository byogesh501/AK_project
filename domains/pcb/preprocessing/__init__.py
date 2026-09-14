"""
PCB Preprocessing

PCB-specific image preprocessing: board cropping, component
segmentation, color normalization, and domain-specific augmentations.
"""
from .annotation import parse_deeppcb_annotation
from .collate import pcb_collate_fn
from .dataset import DeepPCBDataset
from .preprocessor import PCBPreprocessor
from .transforms import TrainTransform, ValTestTransform
