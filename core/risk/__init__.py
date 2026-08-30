"""
Platform Risk & Confidence Module

Risk scoring, confidence calibration, and threshold
management shared across all product domains.
"""

from core.risk.base import BaseRiskScorer, RiskAssessment

__all__ = ["BaseRiskScorer", "RiskAssessment"]
