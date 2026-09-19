
"""Evaluation utilities for the from-scratch PCB detector."""

from typing import Dict, List

import torch


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for boxes in normalized [x1, y1, x2, y2] format."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]),
            dtype=torch.float32,
            device=boxes1.device,
        )

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)

    intersection = wh[..., 0] * wh[..., 1]

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    union = area1[:, None] + area2[None, :] - intersection

    return intersection / (union + 1e-6)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert normalized [xc, yc, w, h] to [x1, y1, x2, y2]."""
    half = boxes[:, 2:] / 2
    return torch.cat(
        (boxes[:, :2] - half, boxes[:, :2] + half),
        dim=1,
    ).clamp(0.0, 1.0)


def match_detections(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Dict[str, object]:
    """
    Match predictions to ground truth using class-aware greedy IoU matching.

    predictions: [N, 6] = x1,y1,x2,y2,confidence,class_id
    targets: [M, 5] = class_id,xc,yc,w,h
    """
    num_predictions = len(predictions)
    num_targets = len(targets)

    if num_predictions == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": num_targets,
            "matched_ious": [],
        }

    if num_targets == 0:
        return {
            "tp": 0,
            "fp": num_predictions,
            "fn": 0,
            "matched_ious": [],
        }

    target_classes = targets[:, 0].long()
    target_boxes = xywh_to_xyxy(targets[:, 1:])

    prediction_classes = predictions[:, 5].long()
    prediction_boxes = predictions[:, :4]

    order = torch.argsort(predictions[:, 4], descending=True)
    matched_targets = set()
    matched_ious: List[float] = []

    tp = 0

    for prediction_index in order.tolist():
        cls = prediction_classes[prediction_index]

        candidates = torch.where(target_classes == cls)[0].tolist()
        candidates = [
            index for index in candidates
            if index not in matched_targets
        ]

        if not candidates:
            continue

        candidate_tensor = torch.tensor(
            candidates,
            device=target_boxes.device,
            dtype=torch.long,
        )

        ious = box_iou_xyxy(
            prediction_boxes[prediction_index:prediction_index + 1],
            target_boxes[candidate_tensor],
        )[0]

        best_position = int(torch.argmax(ious))
        best_iou = float(ious[best_position])
        best_target = candidates[best_position]

        if best_iou >= iou_threshold:
            matched_targets.add(best_target)
            tp += 1
            matched_ious.append(best_iou)

    fp = num_predictions - tp
    fn = num_targets - tp

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_ious": matched_ious,
    }


def evaluate_test_set(
    model,
    dataloader,
    device: torch.device,
    conf_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
    match_iou_threshold: float = 0.5,
    num_classes: int = 6,
) -> Dict[str, object]:
    """Evaluate a detector over a complete dataloader."""
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_matched_ious: List[float] = []

    class_stats = {
        cls: {"tp": 0, "fp": 0, "fn": 0}
        for cls in range(num_classes)
    }

    with torch.no_grad():
        for batch in dataloader:
            images = batch["defect_images"].to(device)
            predictions = model.predict(
                images,
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold,
            )

            for batch_index, prediction in enumerate(predictions):
                valid_targets = batch["annotations"][batch_index][
                    batch["anno_mask"][batch_index]
                ].to(device)

                result = match_detections(
                    prediction,
                    valid_targets,
                    iou_threshold=match_iou_threshold,
                )

                total_tp += result["tp"]
                total_fp += result["fp"]
                total_fn += result["fn"]
                all_matched_ious.extend(result["matched_ious"])

                # Per-class matching.
                prediction_classes = (
                    prediction[:, 5].long()
                    if len(prediction)
                    else torch.empty(0, dtype=torch.long, device=device)
                )
                target_classes = valid_targets[:, 0].long()

                for cls in range(num_classes):
                    cls_predictions = (
                        prediction[prediction_classes == cls]
                        if len(prediction)
                        else prediction
                    )
                    cls_targets = valid_targets[target_classes == cls]

                    cls_result = match_detections(
                        cls_predictions,
                        cls_targets,
                        iou_threshold=match_iou_threshold,
                    )

                    class_stats[cls]["tp"] += cls_result["tp"]
                    class_stats[cls]["fp"] += cls_result["fp"]
                    class_stats[cls]["fn"] += cls_result["fn"]

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = (
        2 * precision * recall / max(precision + recall, 1e-12)
    )

    per_class = {}

    for cls, stats in class_stats.items():
        cls_precision = stats["tp"] / max(
            stats["tp"] + stats["fp"], 1
        )
        cls_recall = stats["tp"] / max(
            stats["tp"] + stats["fn"], 1
        )
        cls_f1 = (
            2 * cls_precision * cls_recall
            / max(cls_precision + cls_recall, 1e-12)
        )

        per_class[cls] = {
            **stats,
            "precision": cls_precision,
            "recall": cls_recall,
            "f1": cls_f1,
        }

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": (
            sum(all_matched_ious) / len(all_matched_ious)
            if all_matched_ious
            else 0.0
        ),
        "per_class": per_class,
    }


def _compute_ap(
    predictions,
    targets_by_image,
    iou_threshold,
):
    """
    Compute AP for one class while keeping predictions and ground truth
    associated with their original images.
    """
    total_targets = sum(len(targets) for targets in targets_by_image)

    if total_targets == 0:
        return 0.0

    # predictions: [(confidence, image_index, box)]
    predictions = sorted(
        predictions,
        key=lambda item: item[0],
        reverse=True,
    )

    matched = {
        image_index: set()
        for image_index in range(len(targets_by_image))
    }

    tp = []
    fp = []

    for confidence, image_index, box in predictions:
        targets = targets_by_image[image_index]

        if len(targets) == 0:
            tp.append(0)
            fp.append(1)
            continue

        target_boxes = xywh_to_xyxy(targets[:, 1:])

        box_tensor = torch.tensor(
            box,
            dtype=target_boxes.dtype,
            device=target_boxes.device,
        ).unsqueeze(0)

        ious = box_iou_xyxy(box_tensor, target_boxes)[0]

        # Only unmatched targets can be selected.
        available = [
            index
            for index in range(len(targets))
            if index not in matched[image_index]
        ]

        if not available:
            tp.append(0)
            fp.append(1)
            continue

        available_tensor = torch.tensor(
            available,
            dtype=torch.long,
            device=target_boxes.device,
        )

        available_ious = ious[available_tensor]

        best_position = int(torch.argmax(available_ious))
        best_iou = float(available_ious[best_position])
        best_target = available[best_position]

        if best_iou >= iou_threshold:
            matched[image_index].add(best_target)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp = torch.tensor(tp, dtype=torch.float64)
    fp = torch.tensor(fp, dtype=torch.float64)

    cumulative_tp = torch.cumsum(tp, dim=0)
    cumulative_fp = torch.cumsum(fp, dim=0)

    precision = cumulative_tp / torch.clamp(
        cumulative_tp + cumulative_fp,
        min=1e-12,
    )

    recall = cumulative_tp / total_targets

    # Precision envelope.
    precision_envelope = torch.flip(
        torch.cummax(
            torch.flip(precision, dims=[0]),
            dim=0,
        ).values,
        dims=[0],
    )

    # Add boundary points and integrate the precision-recall curve.
    recall_points = torch.cat(
        (
            torch.tensor([0.0], dtype=torch.float64),
            recall,
            torch.tensor([1.0], dtype=torch.float64),
        )
    )

    precision_points = torch.cat(
        (
            torch.tensor([1.0], dtype=torch.float64),
            precision_envelope,
            torch.tensor([0.0], dtype=torch.float64),
        )
    )

    return float(
        torch.sum(
            (recall_points[1:] - recall_points[:-1])
            * precision_points[1:]
        )
    )


def evaluate_map(
    model,
    dataloader,
    device,
    num_classes=6,
    conf_threshold=0.001,
    nms_iou_threshold=0.5,
):
    """
    Calculate image-aware mAP@0.5 and mAP@0.5:0.95.

    Predictions are retained with their image index so that a prediction
    can only match ground-truth annotations from the same test image.
    """
    model.eval()

    # One record per image:
    # {
    #   "predictions": tensor[N, 6],
    #   "targets": tensor[M, 5]
    # }
    image_records = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["defect_images"].to(device)

            predictions = model.predict(
                images,
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold,
            )

            for batch_index, prediction in enumerate(predictions):
                valid_targets = batch["annotations"][batch_index][
                    batch["anno_mask"][batch_index]
                ].detach().cpu()

                image_records.append(
                    {
                        "predictions": prediction.detach().cpu(),
                        "targets": valid_targets,
                    }
                )

    def calculate_map(iou_threshold):
        class_aps = []

        for cls in range(num_classes):
            class_predictions = []
            targets_by_image = []

            for image_index, record in enumerate(image_records):
                prediction = record["predictions"]
                targets = record["targets"]

                if len(prediction):
                    class_mask = prediction[:, 5].long() == cls
                    class_prediction = prediction[class_mask]

                    for detection in class_prediction:
                        class_predictions.append(
                            (
                                float(detection[4]),
                                image_index,
                                detection[:4].tolist(),
                            )
                        )

                if len(targets):
                    target_mask = targets[:, 0].long() == cls
                    class_targets = targets[target_mask]
                else:
                    class_targets = torch.empty(
                        (0, 5),
                        dtype=torch.float32,
                    )

                targets_by_image.append(class_targets)

            total_targets = sum(
                len(targets)
                for targets in targets_by_image
            )

            if total_targets == 0:
                continue

            class_ap = _compute_ap(
                class_predictions,
                targets_by_image,
                iou_threshold,
            )

            class_aps.append(class_ap)

        return (
            sum(class_aps) / len(class_aps)
            if class_aps
            else 0.0
        )

    map50 = calculate_map(0.50)

    thresholds = [
        0.50 + 0.05 * index
        for index in range(10)
    ]

    threshold_aps = {
        threshold: calculate_map(threshold)
        for threshold in thresholds
    }

    return {
        "mAP@0.5": map50,
        "mAP@0.5:0.95": sum(threshold_aps.values()) / len(threshold_aps),
        "AP@0.5:0.95 thresholds": threshold_aps,
    }
