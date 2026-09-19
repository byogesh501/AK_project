"""Evaluation helpers kept separate from anomaly model training."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from domains.pcb.anomaly.model import ConvAutoencoder
from domains.pcb.anomaly.scoring import score_pair_inputs


@dataclass(frozen=True)
class ThresholdCalibration:
    """A validation-derived threshold, or its documented unavailability."""

    threshold: Optional[float]
    normal_count: int
    quantile: float
    limitation: Optional[str] = None


def calibrate_validation_threshold(
    image_scores: torch.Tensor,
    normal_mask: Optional[torch.Tensor],
    *,
    split: str = "val",
    quantile: float = 0.99,
) -> ThresholdCalibration:
    """Calibrate from explicitly normal validation examples only.

    DeepPCB's known-defect annotations are not unknown-anomaly labels.  When
    validation data has no known-normal examples, no scientific threshold is
    returned rather than deriving one from known defects.
    """
    if split != "val":
        raise ValueError("Threshold calibration is limited to the validation split")
    if image_scores.ndim != 1:
        raise ValueError("image_scores must have shape [B]")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if normal_mask is None:
        return ThresholdCalibration(
            threshold=None,
            normal_count=0,
            quantile=quantile,
            limitation="No explicit normal validation labels were supplied.",
        )
    if normal_mask.shape != image_scores.shape or normal_mask.dtype != torch.bool:
        raise ValueError("normal_mask must be a boolean tensor with shape [B]")

    normal_scores = image_scores[normal_mask]
    if not len(normal_scores):
        return ThresholdCalibration(
            threshold=None,
            normal_count=0,
            quantile=quantile,
            limitation="Validation data contains no explicitly normal examples.",
        )
    return ThresholdCalibration(
        threshold=float(torch.quantile(normal_scores, quantile)),
        normal_count=len(normal_scores),
        quantile=quantile,
    )


def apply_threshold(image_scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Mark images whose score is at or above a calibrated threshold."""
    if image_scores.ndim != 1:
        raise ValueError("image_scores must have shape [B]")
    return image_scores >= threshold


def score_labeled_batch(
    model: ConvAutoencoder, batch: Dict[str, object]
) -> Dict[str, object]:
    """Score a labeled DeepPCB batch without treating known defects as unknowns.

    ``image_scores`` is for image-level analysis; ``error_maps`` is retained
    separately for later localization comparison with the paired annotations.
    """
    result: Dict[str, object] = score_pair_inputs(model, batch["pair_inputs"])
    result["pair_ids"] = batch["pair_ids"]
    result["annotations"] = batch["annotations"]
    return result
