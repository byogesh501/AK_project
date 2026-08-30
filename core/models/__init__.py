"""
Platform Models Module

Base model architectures, training loops, and model utilities.
All models are built from scratch — no pretrained weights.

PROJECT POLICY: No pretrained models or pretrained weights.
Every model architecture is defined and trained from scratch.
"""

from core.models.base import BaseModel, BaseTrainer

__all__ = ["BaseModel", "BaseTrainer"]
