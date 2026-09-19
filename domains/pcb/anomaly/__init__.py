"""
PCB Anomaly Detection

PCB-specific anomaly detection: solder defects, missing components,
misalignment, and other board-level anomalies.
"""

from .dataset import AnomalyPairTransform, DeepPCBAnomalyDataset, anomaly_collate_fn
from .model import ConvAutoencoder, PAIR_INPUT_CHANNELS, build_pair_difference_input
from .training import AnomalyTrainer, AnomalyTrainingConfig

__all__ = [
    "AnomalyPairTransform",
    "AnomalyTrainer",
    "AnomalyTrainingConfig",
    "ConvAutoencoder",
    "DeepPCBAnomalyDataset",
    "PAIR_INPUT_CHANNELS",
    "anomaly_collate_fn",
    "build_pair_difference_input",
]
