"""Anomaly scores from a ConvAE reconstruction of an aligned PCB pair."""

from typing import Dict

import torch

from domains.pcb.anomaly.model import ConvAutoencoder, MODEL_INPUT_CHANNELS, PAIR_INPUT_CHANNELS


def reconstruction_error_map(
    defect_images: torch.Tensor, reconstructions: torch.Tensor
) -> torch.Tensor:
    """Return channel-mean squared error maps, one ``[H, W]`` map per PCB.

    The autoencoder reconstructs the golden template.  Comparing that expected
    normal appearance with the inspected image produces the anomaly map.
    """
    if defect_images.shape != reconstructions.shape:
        raise ValueError("Defect images and reconstructions must have identical shapes")
    if defect_images.ndim != 4 or defect_images.shape[1] != 3:
        raise ValueError("Defect images and reconstructions must have shape [B, 3, H, W]")
    return (defect_images - reconstructions).square().mean(dim=1)


def image_anomaly_scores(error_maps: torch.Tensor) -> torch.Tensor:
    """Reduce spatial error maps to one scalar score per image."""
    if error_maps.ndim != 3:
        raise ValueError("error_maps must have shape [B, H, W]")
    return error_maps.mean(dim=(1, 2))


def score_model_inputs(model: ConvAutoencoder, model_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Run the ConvAE on RGB inputs and return scores plus error maps."""
    if model_inputs.ndim != 4 or model_inputs.shape[1] != MODEL_INPUT_CHANNELS:
        raise ValueError("model_inputs must have shape [B, 3, H, W]")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        reconstructions = model(model_inputs)
    model.train(was_training)

    error_maps = reconstruction_error_map(model_inputs, reconstructions)
    return {
        "reconstructions": reconstructions,
        "error_maps": error_maps,
        "image_scores": image_anomaly_scores(error_maps),
    }


def score_pair_inputs(model: ConvAutoencoder, pair_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Score RGB model inputs or legacy 9-channel labeled pair inputs."""
    if pair_inputs.ndim != 4 or pair_inputs.shape[1] not in (MODEL_INPUT_CHANNELS, PAIR_INPUT_CHANNELS):
        raise ValueError("pair_inputs must have shape [B, 3, H, W] or [B, 9, H, W]")
    model_inputs = pair_inputs[:, :MODEL_INPUT_CHANNELS]
    return score_model_inputs(model, model_inputs)
