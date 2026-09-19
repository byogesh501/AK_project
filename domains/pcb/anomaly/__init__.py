"""
PCB Anomaly Detection

PCB-specific anomaly detection: solder defects, missing components,
misalignment, and other board-level anomalies.
"""

from .dataset import (
    AnomalyPairTransform,
    DeepPCBAnomalyDataset,
    DeepPCBTemplateAnomalyDataset,
    SyntheticNormalTransform,
    anomaly_collate_fn,
)
from .evaluation import (
    ThresholdCalibration,
    apply_threshold,
    calibrate_validation_threshold,
    score_labeled_batch,
)
from .model import (
    ConvAutoencoder,
    MODEL_INPUT_CHANNELS,
    PAIR_INPUT_CHANNELS,
    build_pair_difference_input,
)
from .scoring import (
    image_anomaly_scores,
    reconstruction_error_map,
    score_model_inputs,
    score_pair_inputs,
)
from .training import AnomalyTrainer, AnomalyTrainingConfig

__all__ = [
    "AnomalyPairTransform",
    "AnomalyTrainer",
    "AnomalyTrainingConfig",
    "ConvAutoencoder",
    "DeepPCBAnomalyDataset",
    "DeepPCBTemplateAnomalyDataset",
    "MODEL_INPUT_CHANNELS",
    "PAIR_INPUT_CHANNELS",
    "SyntheticNormalTransform",
    "ThresholdCalibration",
    "apply_threshold",
    "anomaly_collate_fn",
    "build_pair_difference_input",
    "calibrate_validation_threshold",
    "image_anomaly_scores",
    "reconstruction_error_map",
    "score_labeled_batch",
    "score_model_inputs",
    "score_pair_inputs",
]
