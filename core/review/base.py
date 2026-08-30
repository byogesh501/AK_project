"""
Base interfaces for the human-in-the-loop review pipeline.

These ABCs define the contracts that domain-specific review
implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Data classes ──────────────────────────────────────────


class VerdictStatus(Enum):
    """Possible outcomes of a human review."""

    ACCEPTED = "accepted"          # Model prediction confirmed correct
    REJECTED = "rejected"          # Model prediction marked wrong
    CORRECTED = "corrected"        # Operator supplied the right answer
    NEEDS_REVIEW = "needs_review"  # Flagged for further expert review


@dataclass
class Verdict:
    """A single human verdict on one model prediction."""

    prediction_id: str
    status: VerdictStatus
    corrected_label: Optional[str] = None
    reviewer: str = ""
    notes: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackRecord:
    """Aggregated feedback ready for retraining or analysis."""

    domain: str
    verdicts: List[Verdict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


# ── Abstract base classes ─────────────────────────────────


class BaseReviewStore(ABC):
    """Persist and retrieve human verdicts and corrections."""

    @abstractmethod
    def save_verdict(self, verdict: Verdict) -> None:
        """Store a single verdict."""

    @abstractmethod
    def get_verdicts(
        self,
        prediction_id: Optional[str] = None,
        status: Optional[VerdictStatus] = None,
    ) -> List[Verdict]:
        """Retrieve verdicts, optionally filtered."""

    @abstractmethod
    def export_feedback(self, domain: str) -> FeedbackRecord:
        """Export all verdicts for a domain as a FeedbackRecord."""


class BaseReviewPolicy(ABC):
    """Decide which predictions require human review."""

    @abstractmethod
    def should_review(
        self,
        confidence: float,
        anomaly_score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Return True if this prediction should be routed to a human."""
