"""
Tests for platform base interfaces.

Verifies that ABCs exist and cannot be instantiated directly.
"""

import pytest

from core.preprocessing.base import BasePreprocessor
from core.models.base import BaseModel, BaseTrainer
from core.anomaly.base import BaseAnomalyDetector, AnomalyResult
from core.risk.base import BaseRiskScorer, RiskAssessment
from core.explainability.base import BaseExplainer, Explanation
from core.review.base import (
    BaseReviewStore,
    BaseReviewPolicy,
    Verdict,
    VerdictStatus,
    FeedbackRecord,
)


class TestBasePreprocessor:
    """BasePreprocessor cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BasePreprocessor()  # type: ignore[abstract]


class TestBaseModel:
    """BaseModel cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseModel()  # type: ignore[abstract]


class TestBaseTrainer:
    """BaseTrainer cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseTrainer()  # type: ignore[abstract]


class TestBaseAnomalyDetector:
    """BaseAnomalyDetector cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseAnomalyDetector()  # type: ignore[abstract]

    def test_anomaly_result_defaults(self):
        result = AnomalyResult(
            score=0.85, is_anomaly=True, threshold=0.5, metadata={}
        )
        assert result.score == 0.85
        assert result.is_anomaly is True


class TestBaseRiskScorer:
    """BaseRiskScorer cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseRiskScorer()  # type: ignore[abstract]

    def test_risk_assessment_defaults(self):
        assessment = RiskAssessment(
            risk_score=0.7, confidence=0.9, severity="high", metadata={}
        )
        assert assessment.severity == "high"


class TestBaseExplainer:
    """BaseExplainer cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseExplainer()  # type: ignore[abstract]


class TestBaseReviewStore:
    """BaseReviewStore cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseReviewStore()  # type: ignore[abstract]


class TestBaseReviewPolicy:
    """BaseReviewPolicy cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseReviewPolicy()  # type: ignore[abstract]


class TestVerdictDataClass:
    """Verdict data class can be created with required fields."""

    def test_create_accepted_verdict(self):
        v = Verdict(prediction_id="pred-001", status=VerdictStatus.ACCEPTED)
        assert v.status == VerdictStatus.ACCEPTED
        assert v.corrected_label is None

    def test_create_corrected_verdict(self):
        v = Verdict(
            prediction_id="pred-002",
            status=VerdictStatus.CORRECTED,
            corrected_label="missing_resistor",
            reviewer="inspector-1",
        )
        assert v.corrected_label == "missing_resistor"


class TestFeedbackRecord:
    """FeedbackRecord bundles verdicts for a domain."""

    def test_empty_feedback(self):
        fb = FeedbackRecord(domain="pcb")
        assert fb.domain == "pcb"
        assert fb.verdicts == []
