"""
Platform Anomaly Detection Module

Base anomaly detection algorithms and scoring logic
shared across all product domains.
"""

from core.anomaly.base import AnomalyResult, BaseAnomalyDetector

__all__ = ["AnomalyResult", "BaseAnomalyDetector"]
