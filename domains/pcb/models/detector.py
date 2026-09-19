"""
End-to-End PCB Defect Detector Architecture.
"""
import torch
import torch.nn as nn
import torchvision
from core.models.base import BaseModel

from .backbone import PCBBackbone
from .neck import PCBNeck
from .head import PCBHead

class PCBDetector(BaseModel, nn.Module):
    """
    Compact anchor-free YOLO-style CNN.
    """
    def __init__(self, in_channels=3, base_channels=32, num_classes=6):
        nn.Module.__init__(self)  # Init nn.Module FIRST
        BaseModel.__init__(self)  # Then BaseModel

        self.backbone = PCBBackbone(in_channels=in_channels, base_channels=base_channels)
        self.neck = PCBNeck(in_channels=self.backbone.out_channels, out_channels=base_channels * 4)

        # 3 heads for the 3 scales
        self.heads = nn.ModuleList([
            PCBHead(num_classes, base_channels * 4) for _ in range(3)
        ])

    def forward(self, x):
        """
        Forward pass yielding multi-scale feature maps.
        Returns:
            list of tensors [B, 5 + num_classes, H, W] for strides 8, 16, 32.
        """
        features = self.backbone(x)
        features = self.neck(features)
        out = [head(f) for head, f in zip(self.heads, features)]
        return out

    def predict(self, x, conf_threshold=0.25, nms_iou_threshold=0.5, max_detections=300):
        """Decode multi-scale predictions into normalized detections.

        Returns a list per image. Each detection is:
        [x1, y1, x2, y2, confidence, class_id].
        """
        predictions = self.forward(x)

        strides = [8, 16, 32]
        batch_size = x.shape[0]
        decoded = [[] for _ in range(batch_size)]

        for pred, stride in zip(predictions, strides):
            _, _, height, width = pred.shape

            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=pred.device),
                torch.arange(width, device=pred.device),
                indexing="ij",
            )

            values = pred.permute(0, 2, 3, 1)

            dx = values[..., 0].sigmoid()
            dy = values[..., 1].sigmoid()
            w = values[..., 2].sigmoid()
            h = values[..., 3].sigmoid()

            cx = (grid_x.to(pred.dtype) + dx) / width
            cy = (grid_y.to(pred.dtype) + dy) / height

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            objectness = values[..., 4].sigmoid()
            class_probs = values[..., 5:].sigmoid()

            class_scores, class_ids = class_probs.max(dim=-1)
            confidence = objectness * class_scores

            keep = confidence >= conf_threshold

            for batch_index in range(batch_size):
                mask = keep[batch_index]
                if mask.any():
                    boxes = torch.stack(
                        (
                            x1[batch_index][mask],
                            y1[batch_index][mask],
                            x2[batch_index][mask],
                            y2[batch_index][mask],
                        ),
                        dim=1,
                    )

                    detections = torch.cat(
                        (
                            boxes,
                            confidence[batch_index][mask, None],
                            class_ids[batch_index][mask, None].to(pred.dtype),
                        ),
                        dim=1,
                    )

                    # Keep coordinates inside the normalized image bounds.
                    detections[:, :4] = detections[:, :4].clamp(0.0, 1.0)

                    # Class-aware NMS: boxes from different classes do not suppress
                    # each other.
                    class_offsets = detections[:, 5:6] * 2.0
                    nms_boxes = detections[:, :4] + torch.cat(
                        (class_offsets, class_offsets, class_offsets, class_offsets),
                        dim=1,
                    )

                    keep_indices = torchvision.ops.nms(
                        nms_boxes,
                        detections[:, 4],
                        nms_iou_threshold,
                    )

                    decoded[batch_index].append(detections[keep_indices])

        return [
            torch.cat(items, dim=0) if items else x.new_zeros((0, 6))
            for items in decoded
        ]

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))
