"""
Base interface for model architectures.

Project policy: ALL models are trained from scratch.
Do NOT use pretrained weights (e.g. models with pretrained=True, # noqa: allow-pretrained
weights=..., or downloaded checkpoint files). # noqa: allow-pretrained
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModel(ABC):
    """Contract for all detection/classification models.

    Every model in this project is built from scratch — no pretrained
    weights from external sources.
    """

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Run a forward pass and return predictions."""

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """Run inference on a single input (no grad)."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from disk (our own trained weights only)."""

    def get_config(self) -> Dict[str, Any]:
        """Return the model's architecture configuration."""
        return {}


class BaseTrainer(ABC):
    """Contract for training loops."""

    @abstractmethod
    def train_epoch(self, dataloader: Any) -> Dict[str, float]:
        """Train for one epoch; return metrics dict."""

    @abstractmethod
    def evaluate(self, dataloader: Any) -> Dict[str, float]:
        """Evaluate on a dataset; return metrics dict."""

    @abstractmethod
    def save_checkpoint(self, path: str, epoch: int) -> None:
        """Save a training checkpoint."""

    def load_checkpoint(self, path: str) -> Optional[int]:
        """Load a checkpoint; return the epoch number, or None."""
        return None
