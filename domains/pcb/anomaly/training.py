"""Minimal training loop for the isolated PCB anomaly autoencoder."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

from domains.pcb.training.trainer import resolve_device


@dataclass(frozen=True)
class AnomalyTrainingConfig:
    """Runtime settings for from-scratch autoencoder training."""

    learning_rate: float = 1e-3
    device: str = "auto"


class AnomalyTrainer:
    """Optimize template reconstruction from the aligned pair representation."""

    def __init__(
        self,
        model: nn.Module,
        config: AnomalyTrainingConfig = AnomalyTrainingConfig(),
        criterion: Optional[nn.Module] = None,
    ):
        self.device = resolve_device(config.device)
        self.model = model.to(self.device)
        self.criterion = criterion or nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def _move_batch(self, batch: Dict[str, Any]):
        return (
            batch["pair_inputs"].to(self.device, non_blocking=True),
            batch["normal_targets"].to(self.device, non_blocking=True),
        )

    def _run_epoch(
        self, dataloader: Any, training: bool, max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        was_training = self.model.training
        self.model.train(training)
        total_loss = 0.0
        batches = 0

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch_index, batch in enumerate(dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                pair_inputs, normal_targets = self._move_batch(batch)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                reconstruction = self.model(pair_inputs)
                loss = self.criterion(reconstruction, normal_targets)
                if training:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
                batches += 1

        self.model.train(was_training)
        if not batches:
            raise ValueError("The dataloader produced no batches")
        return {"loss": total_loss / batches, "batches": float(batches)}

    def train_epoch(self, dataloader: Any, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Run one optimization epoch using only golden-template targets."""
        return self._run_epoch(dataloader, training=True, max_batches=max_batches)

    def evaluate(self, dataloader: Any, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Measure validation reconstruction loss without parameter updates."""
        return self._run_epoch(dataloader, training=False, max_batches=max_batches)

    def save_checkpoint(
        self, path: Union[str, Path], epoch: int, best_val_loss: Optional[float] = None
    ) -> None:
        """Save this anomaly component separately from the known-defect detector."""
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
