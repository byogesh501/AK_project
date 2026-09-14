"""Training utilities for the PCB detector."""

from .trainer import PCBTrainer, TrainingConfig, resolve_device

__all__ = ["PCBTrainer", "TrainingConfig", "resolve_device"]
