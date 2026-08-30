"""
Base interface for preprocessing pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BasePreprocessor(ABC):
    """Contract for all domain preprocessing pipelines.

    Subclasses implement domain-specific image loading, resizing,
    normalization, and augmentation logic.
    """

    @abstractmethod
    def load_image(self, path: str) -> Any:
        """Load a single image from disk and return it as a tensor/array."""

    @abstractmethod
    def preprocess(self, image: Any) -> Any:
        """Apply the full preprocessing pipeline to one image."""

    @abstractmethod
    def augment(self, image: Any) -> List[Any]:
        """Return augmented copies of the image for training."""

    def get_config(self) -> Dict[str, Any]:
        """Return the current preprocessing configuration."""
        return {}
