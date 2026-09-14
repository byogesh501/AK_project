"""
Path Aggregation feature Neck (FPN/PAN style) for fusing multi-scale features.
"""
import torch
import torch.nn as nn
from .modules import ConvBNAct

class PCBNeck(nn.Module):
    """
    Lightweight FPN + PANet neck.
    Takes C3, C4, C5 and outputs P3, P4, P5 with identical channel dims.
    """
    def __init__(self, in_channels, out_channels=128):
        # in_channels: [c3, c4, c5]
        super().__init__()
        self.out_channels = out_channels

        # Top-down FPN
        self.lat5 = ConvBNAct(in_channels[2], out_channels, k=1, s=1, p=0)
        self.lat4 = ConvBNAct(in_channels[1], out_channels, k=1, s=1, p=0)
        self.lat3 = ConvBNAct(in_channels[0], out_channels, k=1, s=1, p=0)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.fpn4 = ConvBNAct(out_channels, out_channels, k=3, s=1, p=1)
        self.fpn3 = ConvBNAct(out_channels, out_channels, k=3, s=1, p=1)

        # Bottom-up PAN
        self.down3 = ConvBNAct(out_channels, out_channels, k=3, s=2, p=1)
        self.pan4 = ConvBNAct(out_channels, out_channels, k=3, s=1, p=1)

        self.down4 = ConvBNAct(out_channels, out_channels, k=3, s=2, p=1)
        self.pan5 = ConvBNAct(out_channels, out_channels, k=3, s=1, p=1)

    def forward(self, features):
        c3, c4, c5 = features

        # FPN Top-Down
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + self.upsample(p5)
        p4 = self.fpn4(p4)

        p3 = self.lat3(c3) + self.upsample(p4)
        p3 = self.fpn3(p3)

        # PAN Bottom-Up
        p4_pan = self.pan4(p4 + self.down3(p3))
        p5_pan = self.pan5(p5 + self.down4(p4_pan))

        return [p3, p4_pan, p5_pan]
