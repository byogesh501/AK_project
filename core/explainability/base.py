"""
Base interface for model explainability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Explanation:
    """Visual explanation for a single prediction."""

    heatmap: Any                    # Attribution map (numpy array or tensor)
    predicted_label: str
    confidence: float
    method: str                     # e.g. "grad-cam", "saliency"
    metadata: Dict[str, Any]


class BaseExplainer(ABC):
    """Contract for visual explanation generators."""

    @abstractmethod
    def explain(self, model: Any, x: Any) -> Explanation:
        """Generate a visual explanation for a single prediction."""

    @abstractmethod
    def render(
        self,
        explanation: Explanation,
        original_image: Any,
        output_path: Optional[str] = None,
    ) -> Any:
        """Overlay the explanation on the original image.

        Returns the composited image. If output_path is given,
        also saves it to disk.
        """

    def get_config(self) -> Dict[str, Any]:
        """Return the explainer's configuration."""
        return {}
