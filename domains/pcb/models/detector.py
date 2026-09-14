"""
End-to-End PCB Defect Detector Architecture.
"""
import torch
import torch.nn as nn
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

    def predict(self, x):
        pass

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))
