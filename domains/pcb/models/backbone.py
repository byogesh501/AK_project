"""
Lightweight Residual Backbone for feature extraction.
Produces multi-scale features at strides 8, 16, and 32.
"""
import torch.nn as nn
from .modules import ConvBNAct, ResBlock

class PCBBackbone(nn.Module):
    """
    Compact CNN Backbone extracting features at strides 8, 16, 32.
    """
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.out_channels = [base_channels * 4, base_channels * 8, base_channels * 16]

        # Stem: stride 2 -> 320x320
        self.stem = ConvBNAct(in_channels, base_channels, k=3, s=2, p=1)

        # Layer 1: stride 2 -> 160x160
        self.layer1 = self._make_layer(base_channels, base_channels * 2, num_blocks=2, stride=2)

        # Layer 2: stride 2 -> 80x80 (P3, stride 8)
        self.layer2 = self._make_layer(base_channels * 2, self.out_channels[0], num_blocks=2, stride=2)

        # Layer 3: stride 2 -> 40x40 (P4, stride 16)
        self.layer3 = self._make_layer(self.out_channels[0], self.out_channels[1], num_blocks=2, stride=2)

        # Layer 4: stride 2 -> 20x20 (P5, stride 32)
        self.layer4 = self._make_layer(self.out_channels[1], self.out_channels[2], num_blocks=2, stride=2)

    def _make_layer(self, in_c, out_c, num_blocks, stride):
        layers = []
        layers.append(ResBlock(in_c, out_c, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return [c3, c4, c5]
