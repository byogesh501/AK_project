"""
Common CNN modules (Conv-BN-Act, Residual Blocks) for the detector.
"""
import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    """Standard Convolution + BatchNorm + Activation block."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1, groups=1, act="silu"):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    """Standard Residual Block."""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        # 1x1 squeeze then 3x3 if needed, or straight 3x3s
        # We will use two 3x3 convs for simplicity
        self.conv1 = ConvBNAct(in_c, out_c, k=3, s=stride, p=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )

        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_c)
            )

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return self.act(out)
