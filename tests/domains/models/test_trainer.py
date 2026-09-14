"""Focused tests for the minimal PCB train/validation/checkpoint loop."""

import torch
import torch.nn as nn

from domains.pcb.models import PCBLoss
from domains.pcb.training import PCBTrainer, TrainingConfig


class TinyDetector(nn.Module):
    """A learnable three-scale output source for trainer behavior tests."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1, 8, 1, 1))

    def forward(self, images):
        batch_size = images.shape[0]
        return [
            self.logits.expand(batch_size, -1, 8, 8),
            self.logits.expand(batch_size, -1, 4, 4),
            self.logits.expand(batch_size, -1, 2, 2),
        ]


def _batch():
    return {
        "defect_images": torch.randn(2, 3, 64, 64),
        "annotations": torch.tensor(
            [
                [[0, 0.5625, 0.5625, 0.125, 0.125]],
                [[1, 0.1875, 0.1875, 0.125, 0.125]],
            ],
            dtype=torch.float32,
        ),
        "anno_mask": torch.tensor([[True], [True]]),
    }


def _trainer():
    return PCBTrainer(
        TinyDetector(),
        TrainingConfig(learning_rate=1e-2, device="cpu"),
        criterion=PCBLoss(num_classes=3),
    )


def test_train_epoch_updates_parameters_and_reports_losses():
    trainer = _trainer()
    before = trainer.model.logits.detach().clone()

    metrics = trainer.train_epoch([_batch(), _batch()])

    assert metrics["batches"] == 2
    assert metrics["first_loss"] > 0
    assert metrics["last_loss"] > 0
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(before, trainer.model.logits.detach())


def test_validation_runs_without_gradients_or_parameter_updates():
    trainer = _trainer()
    before = trainer.model.logits.detach().clone()
    trainer.model.train()

    metrics = trainer.evaluate([_batch()])

    assert metrics["batches"] == 1
    assert trainer.model.training
    assert trainer.model.logits.grad is None
    assert torch.equal(before, trainer.model.logits.detach())


def test_checkpoint_is_written(tmp_path):
    trainer = _trainer()
    checkpoint_path = tmp_path / "best.pt"

    trainer.save_checkpoint(checkpoint_path, epoch=1, best_val_loss=0.5)

    assert checkpoint_path.exists()
