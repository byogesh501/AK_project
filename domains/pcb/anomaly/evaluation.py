"""Evaluation helpers kept separate from anomaly model training."""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping, Optional, Sequence

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


def annotation_mask(
    annotations: Sequence[Mapping[str, float]], height: int, width: int
) -> torch.Tensor:
    """Rasterize normalized DeepPCB boxes into one boolean image mask."""
    mask = torch.zeros((height, width), dtype=torch.bool)
    for annotation in annotations:
        x_center = float(annotation["x_center"]) * width
        y_center = float(annotation["y_center"]) * height
        box_width = float(annotation["width"]) * width
        box_height = float(annotation["height"]) * height
        x1 = max(0, min(width, math.floor(x_center - box_width / 2)))
        y1 = max(0, min(height, math.floor(y_center - box_height / 2)))
        x2 = max(x1, min(width, math.ceil(x_center + box_width / 2)))
        y2 = max(y1, min(height, math.ceil(y_center + box_height / 2)))
        mask[y1:y2, x1:x2] = True
    return mask


def score_statistics(scores: torch.Tensor) -> Dict[str, float]:
    """Return compact descriptive statistics for a score vector."""
    if scores.ndim != 1 or not len(scores):
        raise ValueError("scores must be a non-empty tensor with shape [N]")
    scores = scores.float()
    return {
        "count": float(len(scores)),
        "mean": float(scores.mean()),
        "std": float(scores.std(unbiased=False)),
        "min": float(scores.min()),
        "median": float(scores.median()),
        "max": float(scores.max()),
    }


def known_defect_metrics(
    image_scores: torch.Tensor,
    annotations: Sequence[Sequence[Mapping[str, float]]],
    threshold: float,
) -> Dict[str, object]:
    """Evaluate image-level thresholding against known DeepPCB boxes.

    This is a known-defect experiment only.  DeepPCB supplies no unknown or
    defect-free inspection labels, so the metrics do not estimate unknown-
    anomaly performance.
    """
    if image_scores.ndim != 1 or len(image_scores) != len(annotations):
        raise ValueError("image_scores and annotations must describe the same batch")
    labels = torch.tensor([bool(items) for items in annotations], dtype=torch.bool)
    predictions = apply_threshold(image_scores, threshold)
    tp = int((predictions & labels).sum())
    fp = int((predictions & ~labels).sum())
    tn = int((~predictions & ~labels).sum())
    fn = int((~predictions & labels).sum())
    total = len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "count": total,
        "positive_count": int(labels.sum()),
        "negative_count": int((~labels).sum()),
        "predicted_positive_count": int(predictions.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "known_defect_evaluation_only": True,
    }


def localization_metrics(
    error_maps: torch.Tensor,
    annotations: Sequence[Sequence[Mapping[str, float]]],
) -> Dict[str, float]:
    """Compare error-map peaks and box-region error for known defects."""
    if error_maps.ndim != 3 or len(error_maps) != len(annotations):
        raise ValueError("error_maps and annotations must describe the same batch")
    hits = []
    inside_means = []
    outside_means = []
    for error_map, items in zip(error_maps.float(), annotations):
        if not items:
            continue
        mask = annotation_mask(items, error_map.shape[0], error_map.shape[1])
        if not mask.any():
            continue
        peak = int(error_map.reshape(-1).argmax())
        hits.append(bool(mask.reshape(-1)[peak]))
        inside_means.append(float(error_map[mask].mean()))
        if (~mask).any():
            outside_means.append(float(error_map[~mask].mean()))
    result = {
        "annotated_images": float(len(hits)),
        "top1_peak_hit_rate": sum(hits) / len(hits) if hits else 0.0,
        "mean_inside_error": sum(inside_means) / len(inside_means) if inside_means else 0.0,
        "mean_outside_error": sum(outside_means) / len(outside_means) if outside_means else 0.0,
    }
    result["inside_to_outside_error_ratio"] = (
        result["mean_inside_error"] / result["mean_outside_error"]
        if result["mean_outside_error"]
        else 0.0
    )
    return result


def per_class_known_defect_metrics(
    image_scores: torch.Tensor,
    error_maps: torch.Tensor,
    annotations: Sequence[Sequence[Mapping[str, float]]],
    threshold: float,
    class_names: Optional[Mapping[int, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Report score/flag/localization summaries for each labeled defect class."""
    predictions = apply_threshold(image_scores, threshold)
    output: Dict[str, Dict[str, float]] = {}
    class_ids = sorted({int(item["class_id"]) for items in annotations for item in items})
    for class_id in class_ids:
        selected = [class_id in {int(item["class_id"]) for item in items} for items in annotations]
        indices = [index for index, included in enumerate(selected) if included]
        name = (
            class_names.get(class_id, class_names.get(str(class_id), str(class_id)))
            if class_names
            else str(class_id)
        )
        class_maps = error_maps[indices]
        class_annotations = [annotations[index] for index in indices]
        localization = localization_metrics(class_maps, class_annotations)
        output[name] = {
            "class_id": float(class_id),
            "count": float(len(indices)),
            "mean_score": float(image_scores[indices].mean()),
            "flagged_rate": float(predictions[indices].float().mean()),
            "top1_peak_hit_rate": localization["top1_peak_hit_rate"],
        }
    return output


def evaluate_known_defect_batches(
    model: ConvAutoencoder,
    batches: Iterable[Dict[str, object]],
    threshold: float,
    *,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, object]:
    """Score test batches and return maps plus known-defect summaries."""
    score_parts = []
    map_parts = []
    pair_ids = []
    annotations = []
    for batch_index, batch in enumerate(batches):
        if max_batches is not None and batch_index >= max_batches:
            break
        result = score_labeled_batch(
            model,
            {**batch, "pair_inputs": batch["pair_inputs"].to(device, non_blocking=True)},
        )
        score_parts.append(result["image_scores"].cpu())
        map_parts.append(result["error_maps"].cpu())
        pair_ids.extend(result["pair_ids"])
        annotations.extend(result["annotations"])
    if not score_parts:
        raise ValueError("The test dataloader produced no batches")
    image_scores = torch.cat(score_parts)
    error_maps = torch.cat(map_parts)
    return {
        "pair_ids": pair_ids,
        "annotations": annotations,
        "image_scores": image_scores,
        "error_maps": error_maps,
        "predictions": apply_threshold(image_scores, threshold),
        "metrics": known_defect_metrics(image_scores, annotations, threshold),
        "localization": localization_metrics(error_maps, annotations),
    }
