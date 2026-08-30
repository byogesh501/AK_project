"""
Base interface for risk scoring and confidence calibration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RiskAssessment:
    """Risk assessment for a single prediction."""

    risk_score: float        # Overall risk (0.0–1.0, higher = riskier)
    confidence: float        # Model confidence in prediction (0.0–1.0)
    severity: str            # e.g. "low", "medium", "high", "critical"
    metadata: Dict[str, Any] # Scorer-specific details


class BaseRiskScorer(ABC):
    """Contract for risk scoring and severity classification."""

    @abstractmethod
    def assess(
        self,
        prediction: Any,
        confidence: float,
        anomaly_score: float,
    ) -> RiskAssessment:
        """Produce a risk assessment for a single prediction."""

    @abstractmethod
    def calibrate(self, validation_data: Any) -> None:
        """Calibrate confidence scores on held-out data."""

    def get_config(self) -> Dict[str, Any]:
        """Return the scorer's configuration."""
        return {}
