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
    annotation_mask,
    apply_threshold,
    calibrate_validation_threshold,
    evaluate_known_defect_batches,
    known_defect_metrics,
    localization_metrics,
    per_class_known_defect_metrics,
    score_labeled_batch,
    score_statistics,
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

# Lazy import training components to avoid core/models dependencies
# breaking anomaly evaluation in constrained environments (e.g., Colab tests)

def __getattr__(name):
    if name in ("AnomalyTrainer", "AnomalyTrainingConfig"):
        from .training import AnomalyTrainer, AnomalyTrainingConfig
        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


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
    "annotation_mask",
    "apply_threshold",
    "anomaly_collate_fn",
    "build_pair_difference_input",
    "calibrate_validation_threshold",
    "evaluate_known_defect_batches",
    "image_anomaly_scores",
    "known_defect_metrics",
    "localization_metrics",
    "per_class_known_defect_metrics",
    "reconstruction_error_map",
    "score_labeled_batch",
    "score_model_inputs",
    "score_pair_inputs",
    "score_statistics",
]
