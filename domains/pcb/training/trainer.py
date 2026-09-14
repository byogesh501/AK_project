"""Minimal train/validation loop for the from-scratch PCB detector."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from core.models import BaseTrainer
from domains.pcb.models import PCBLoss


@dataclass(frozen=True)
class TrainingConfig:
    """Runtime settings used by the PCB training entry point."""

    learning_rate: float = 1e-3
    device: str = "auto"


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve the supported training device without changing model behavior."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved


class PCBTrainer(BaseTrainer):
    """Small, explicit optimizer loop around ``PCBDetector`` and ``PCBLoss``."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig = TrainingConfig(),
        criterion: Optional[PCBLoss] = None,
    ):
        self.device = resolve_device(config.device)
        self.model = model.to(self.device)
        self.criterion = criterion or PCBLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def _move_batch(self, batch: Dict[str, Any]):
        """Move the detector inputs and collated annotation tensors to the device."""
        return (
            batch["defect_images"].to(self.device, non_blocking=True),
            batch["anno_mask"].to(self.device, non_blocking=True),
            batch["annotations"].to(self.device, non_blocking=True),
        )

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        return {"loss": 0.0, "l_box": 0.0, "l_obj": 0.0, "l_cls": 0.0, "batches": 0.0}

    @staticmethod
    def _summarize(metrics: Dict[str, float], first_loss: Optional[float], last_loss: Optional[float]):
        batches = metrics["batches"]
        if not batches:
            raise ValueError("The dataloader produced no batches")
        for key in ("loss", "l_box", "l_obj", "l_cls"):
            metrics[key] /= batches
        metrics["first_loss"] = first_loss
        metrics["last_loss"] = last_loss
        return metrics

    def train_epoch(self, dataloader: Any, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Train for one pass (or a bounded number of batches) over train data."""
        self.model.train()
        metrics = self._empty_metrics()
        first_loss = None
        last_loss = None

        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images, targets_mask, targets = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss_dict = self.criterion(self.model(images), targets_mask, targets)
            loss_dict["loss"].backward()
            self.optimizer.step()

            loss_value = loss_dict["loss"].item()
            first_loss = loss_value if first_loss is None else first_loss
            last_loss = loss_value
            for key in ("loss", "l_box", "l_obj", "l_cls"):
                metrics[key] += loss_dict[key].item()
            metrics["batches"] += 1

        return self._summarize(metrics, first_loss, last_loss)

    def evaluate(self, dataloader: Any, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Evaluate validation data without gradient calculation or updates."""
        was_training = self.model.training
        self.model.eval()
        metrics = self._empty_metrics()
        first_loss = None
        last_loss = None

        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break

                images, targets_mask, targets = self._move_batch(batch)
                loss_dict = self.criterion(self.model(images), targets_mask, targets)
                loss_value = loss_dict["loss"].item()
                first_loss = loss_value if first_loss is None else first_loss
                last_loss = loss_value
                for key in ("loss", "l_box", "l_obj", "l_cls"):
                    metrics[key] += loss_dict[key].item()
                metrics["batches"] += 1

        self.model.train(was_training)
        return self._summarize(metrics, first_loss, last_loss)

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        best_val_loss: Optional[float] = None,
    ) -> None:
        """Save only this run's model and optimizer state."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            checkpoint_path,
        )
