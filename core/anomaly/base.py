"""
Base interface for anomaly detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AnomalyResult:
    """Result from an anomaly detector for a single input."""

    score: float              # Higher = more anomalous (0.0–1.0)
    is_anomaly: bool          # Above threshold?
    threshold: float          # Decision threshold used
    metadata: Dict[str, Any]  # Detector-specific details


class BaseAnomalyDetector(ABC):
    """Contract for all anomaly detection algorithms."""

    @abstractmethod
    def fit(self, data: Any) -> None:
        """Fit the detector on normal (non-anomalous) data."""

    @abstractmethod
    def score(self, x: Any) -> float:
        """Return an anomaly score for a single input."""

    @abstractmethod
    def detect(self, x: Any) -> AnomalyResult:
        """Score and classify a single input."""

    def set_threshold(self, threshold: float) -> None:
        """Update the decision threshold."""
        self.threshold = threshold

    def get_config(self) -> Dict[str, Any]:
        """Return the detector's configuration."""
        return {}
