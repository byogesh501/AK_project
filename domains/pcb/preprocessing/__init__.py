"""
PCB Preprocessing

PCB-specific image preprocessing: board cropping, component
segmentation, color normalization, and domain-specific augmentations.
"""
from .annotation import parse_deeppcb_annotation
