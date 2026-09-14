"""
Detection heads directly predicting bounding boxes and class logits.
Anchor-free YOLO-style point head.
"""
import torch
import torch.nn as nn
from .modules import ConvBNAct

class PCBHead(nn.Module):
    """
    Decoupled head yielding (bbox, obj, cls).
    Since this is anchor-free, each grid cell predicts ONE box.
    Output shapes per level: [B, 4 + 1 + num_classes, H, W]
    """
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.num_classes = num_classes

        # We decouple the box branch and the class/obj branch
        self.stem_box = ConvBNAct(in_channels, in_channels, k=3, s=1, p=1)
        self.stem_cls = ConvBNAct(in_channels, in_channels, k=3, s=1, p=1)

        self.box_pred = nn.Conv2d(in_channels, 4, 1)        # [dx, dy, tw, th]
        self.obj_pred = nn.Conv2d(in_channels, 1, 1)        # objectness logic
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x):
        feat_box = self.stem_box(x)
        feat_cls = self.stem_cls(x)

        bbox = self.box_pred(feat_box)
        obj = self.obj_pred(feat_cls)
        cls_logits = self.cls_pred(feat_cls)

        # Concatenate on channel dim: [B, 4 + 1 + C, H, W]
        return torch.cat([bbox, obj, cls_logits], dim=1)
