"""
Platform Explainability Module

Visual explanation utilities (e.g., Grad-CAM, saliency maps)
shared across all product domains.
"""

from core.explainability.base import BaseExplainer, Explanation

__all__ = ["BaseExplainer", "Explanation"]
