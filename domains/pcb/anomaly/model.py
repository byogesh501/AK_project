"""Compact from-scratch autoencoder for aligned PCB pair inputs."""

import torch
from torch import nn


PAIR_INPUT_CHANNELS = 9
MODEL_INPUT_CHANNELS = 3


def build_pair_difference_input(
    defect_image: torch.Tensor, template_image: torch.Tensor
) -> torch.Tensor:
    """Build a spatially aligned defect/template/difference representation.

    The first three channels are the inspected PCB, the next three are its
    registered golden template, and the final three are the signed pixel-wise
    difference (``defect - template``).  These are deliberately grouped from
    the same aligned pair rather than unrelated image concatenation.

    Accepts either one image ``[3, H, W]`` or a batch ``[B, 3, H, W]``.
    """
    if defect_image.shape != template_image.shape:
        raise ValueError("Defect and template tensors must have identical shapes")
    if defect_image.ndim not in (3, 4):
        raise ValueError("Pair tensors must have shape [3, H, W] or [B, 3, H, W]")

    channel_dim = 0 if defect_image.ndim == 3 else 1
    if defect_image.shape[channel_dim] != 3:
        raise ValueError("Pair tensors must contain three RGB channels")

    difference = defect_image - template_image
    return torch.cat((defect_image, template_image, difference), dim=channel_dim)


class _ConvBlock(nn.Sequential):
    """Convolution, batch normalization, and ReLU for the encoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class _DeconvBlock(nn.Sequential):
    """Transposed convolution, batch normalization, and ReLU for decoding."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ConvAutoencoder(nn.Module):
    """Small denoising autoencoder for normal RGB PCB appearances.

    The model has no external initialization or pretrained components.  Its
    four downsampling stages require the native DeepPCB 640x640 inputs (or
    another spatial size divisible by 16).  It receives only an RGB image and
    emits a 3-channel image in ``[0, 1]`` suitable for comparison with the
    inspected PCB.  The reconstruction target is never supplied as a channel.
    """

    def __init__(self, base_channels: int = 16, in_channels: int = MODEL_INPUT_CHANNELS):
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if in_channels != MODEL_INPUT_CHANNELS:
            raise ValueError(f"ConvAutoencoder expects {MODEL_INPUT_CHANNELS} input channels")

        self.encoder = nn.Sequential(
            _ConvBlock(in_channels, base_channels),
            _ConvBlock(base_channels, base_channels * 2),
            _ConvBlock(base_channels * 2, base_channels * 4),
            _ConvBlock(base_channels * 4, base_channels * 4),
        )
        self.decoder = nn.Sequential(
            _DeconvBlock(base_channels * 4, base_channels * 2),
            _DeconvBlock(base_channels * 2, base_channels),
            _DeconvBlock(base_channels, base_channels),
            nn.ConvTranspose2d(base_channels, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        """Reconstruct the normal RGB target from an RGB model input."""
        if model_input.ndim != 4:
            raise ValueError("model_input must have shape [B, 3, H, W]")
        if model_input.shape[1] != MODEL_INPUT_CHANNELS:
            raise ValueError(f"model_input must have {MODEL_INPUT_CHANNELS} channels")
        height, width = model_input.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError("Input height and width must be divisible by 16")
        return self.decoder(self.encoder(model_input))
