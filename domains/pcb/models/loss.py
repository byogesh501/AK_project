"""
Mathematical IoU and Loss utilities for the PCB Detector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def box_iou(boxes1, boxes2, CIoU=True):
    """
    Computes Intersection over Union and CIoU.
    boxes1, boxes2: [N, 4] formatted as [x1, y1, x2, y2]
    Returns: iou, ciou penalty term
    """
    # Overlap
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    if not CIoU:
        return iou

    # CIoU penalty
    cw = torch.max(boxes1[:, None, 2], boxes2[:, 2]) - torch.min(boxes1[:, None, 0], boxes2[:, 0])
    ch = torch.max(boxes1[:, None, 3], boxes2[:, 3]) - torch.min(boxes1[:, None, 1], boxes2[:, 1])
    c_dist = cw ** 2 + ch ** 2 + 1e-6

    rho2 = ((boxes1[:, None, 0] + boxes1[:, None, 2] - boxes2[:, 0] - boxes2[:, 2]) ** 2 +
            (boxes1[:, None, 1] + boxes1[:, None, 3] - boxes2[:, 1] - boxes2[:, 3]) ** 2) / 4

    w1, h1 = boxes1[:, 2] - boxes1[:, 0], boxes1[:, 3] - boxes1[:, 1]
    w2, h2 = boxes2[:, 2] - boxes2[:, 0], boxes2[:, 3] - boxes2[:, 1]

    v = (4 / (3.1415926535 ** 2)) * torch.pow(torch.atan(w1 / torch.clamp(h1, min=1e-6))[:, None] - torch.atan(w2 / torch.clamp(h2, min=1e-6)), 2)
    with torch.no_grad():
        alpha = v / torch.clamp(1 - iou + v, min=1e-6)

    ciou = iou - (rho2 / c_dist + v * alpha)
    return iou, ciou


class PCBLoss(nn.Module):
    """
    YOLO-style detection loss optimized for the anchor-free formulation.
    """
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        self.bce_obj = nn.BCEWithLogitsLoss()
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.strides = [8, 16, 32]

    @staticmethod
    def _xywh_to_xyxy(boxes):
        """Convert normalized ``(xc, yc, w, h)`` boxes to ``(x1, y1, x2, y2)``."""
        half_wh = boxes[:, 2:] / 2
        return torch.cat((boxes[:, :2] - half_wh, boxes[:, :2] + half_wh), dim=1)

    @staticmethod
    def _decode_boxes(raw_boxes, grid_x, grid_y, width, height):
        """Decode head ``(dx, dy, tw, th)`` values at their grid locations.

        Centers are offsets within a grid cell and widths/heights are normalized
        to the input image.  Sigmoid keeps each decoded value in its valid
        normalized range while retaining the head's existing raw tensor layout.
        """
        center_x = (grid_x.to(raw_boxes.dtype) + raw_boxes[:, 0].sigmoid()) / width
        center_y = (grid_y.to(raw_boxes.dtype) + raw_boxes[:, 1].sigmoid()) / height
        wh = raw_boxes[:, 2:].sigmoid()
        return torch.stack((center_x, center_y, wh[:, 0], wh[:, 1]), dim=1)

    def _assign_levels(self, boxes, input_width, input_height):
        """Choose the stride-8/16/32 level from the largest box side in pixels."""
        box_side = torch.maximum(boxes[:, 2] * input_width, boxes[:, 3] * input_height)
        # The detector's three FPN levels cover small, medium, and large boxes.
        return torch.where(
            box_side < 64,
            torch.zeros_like(box_side, dtype=torch.long),
            torch.where(
                box_side < 128,
                torch.ones_like(box_side, dtype=torch.long),
                torch.full_like(box_side, 2, dtype=torch.long),
            ),
        )

    def forward(self, predictions, targets_mask, targets):
        """
        predictions: List of Tensors [B, 5+C, H, W] for strides 8,16,32.
        targets: [B, max_annos, 5] (class_id, xc, yc, w, h)
        targets_mask: [B, max_annos] indicates valid padding.
        """
        if len(predictions) != len(self.strides):
            raise ValueError(f"Expected {len(self.strides)} prediction scales, got {len(predictions)}")

        device = predictions[0].device
        if targets_mask.shape != targets.shape[:2]:
            raise ValueError("targets_mask must have shape [B, max_annos] matching targets")

        # Padded target rows never enter any assignment or loss calculation.
        batch_indices, annotation_indices = targets_mask.to(device=device, dtype=torch.bool).nonzero(as_tuple=True)
        valid_targets = targets.to(device=device)[batch_indices, annotation_indices]

        loss_box = predictions[0].sum() * 0
        loss_obj = predictions[0].sum() * 0
        loss_cls = predictions[0].sum() * 0

        # All feature maps describe the same input image; derive its dimensions
        # from the stride-8 map instead of hard-coding 640.
        input_height = predictions[0].shape[2] * self.strides[0]
        input_width = predictions[0].shape[3] * self.strides[0]
        levels = self._assign_levels(valid_targets[:, 1:], input_width, input_height) if len(valid_targets) else None

        box_losses = []
        cls_logits = []
        cls_targets = []

        for level, pred in enumerate(predictions):
            batch_size, channels, height, width = pred.shape
            if channels != 5 + self.num_classes:
                raise ValueError(f"Prediction scale {level} has {channels} channels; expected {5 + self.num_classes}")
            if batch_size != targets.shape[0]:
                raise ValueError("Prediction and target batch sizes must match")

            object_targets = torch.zeros_like(pred[:, 4])
            if len(valid_targets):
                assigned = levels == level
                assigned_batches = batch_indices[assigned]
                assigned_targets = valid_targets[assigned]

                if len(assigned_targets):
                    class_ids = assigned_targets[:, 0].long()
                    if (class_ids < 0).any() or (class_ids >= self.num_classes).any():
                        raise ValueError("Valid target class IDs must be in [0, num_classes)")

                    grid_x = torch.floor(assigned_targets[:, 1] * width).long().clamp_(0, width - 1)
                    grid_y = torch.floor(assigned_targets[:, 2] * height).long().clamp_(0, height - 1)
                    object_targets[assigned_batches, grid_y, grid_x] = 1

                    point_predictions = pred.permute(0, 2, 3, 1)[assigned_batches, grid_y, grid_x]
                    decoded_boxes = self._decode_boxes(point_predictions[:, :4], grid_x, grid_y, width, height)
                    gt_boxes = assigned_targets[:, 1:]
                    _, ciou = box_iou(self._xywh_to_xyxy(decoded_boxes), self._xywh_to_xyxy(gt_boxes))
                    box_losses.append(1 - ciou.diagonal())
                    cls_logits.append(point_predictions[:, 5:])
                    cls_targets.append(F.one_hot(class_ids, self.num_classes).to(point_predictions.dtype))

            # Objectness is supervised at every cell, with positives only at
            # cells selected by non-padded ground-truth annotations.
            loss_obj = loss_obj + self.bce_obj(pred[:, 4], object_targets)

        if box_losses:
            loss_box = torch.cat(box_losses).mean()
            loss_cls = self.bce_cls(torch.cat(cls_logits), torch.cat(cls_targets))

        loss_total = loss_box * 0.05 + loss_obj * 1.0 + loss_cls * 0.5

        return {
            'loss': loss_total,
            'l_box': loss_box.detach(),
            'l_obj': loss_obj.detach(),
            'l_cls': loss_cls.detach()
        }
