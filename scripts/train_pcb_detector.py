"""Train the existing PCB detector from scratch on DeepPCB train/val splits."""

import argparse
import random
from pathlib import Path

import torch

from domains.pcb.config import NUM_CLASSES
from domains.pcb.models import PCBDetector
from domains.pcb.preprocessing.preprocessor import PCBPreprocessor
from domains.pcb.training import PCBTrainer, TrainingConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "pcb_detector_best.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_root = PROJECT_ROOT / "data" / "raw" / "deeppcb"
    processed_root = PROJECT_ROOT / "data" / "processed"
    preprocessor = PCBPreprocessor(
        data_root=str(data_root),
        manifest_path=str(processed_root / "manifest.csv"),
        class_map_path=str(processed_root / "class_map.json"),
    )
    # The current detector is 3-channel and trains on defect_images.  The
    # preprocessor continues to apply paired transforms to defect/template pairs.
    train_loader = preprocessor.create_dataloader(
        "train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu" and torch.cuda.is_available(),
    )
    val_loader = preprocessor.create_dataloader(
        "val",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu" and torch.cuda.is_available(),
    )

    model = PCBDetector(in_channels=3, num_classes=NUM_CLASSES)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainer = PCBTrainer(model, TrainingConfig(args.learning_rate, args.device))
    best_val_loss = float("inf")

    print(f"Device: {trainer.device}")
    print(f"Train samples: {len(train_loader.dataset)}; validation samples: {len(val_loader.dataset)}")
    print(f"Trainable parameters: {parameter_count:,}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(train_loader, max_batches=args.max_train_batches)
        val_metrics = trainer.evaluate(val_loader, max_batches=args.max_val_batches)
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_metrics['loss']:.6f} "
            f"(first={train_metrics['first_loss']:.6f}, last={train_metrics['last_loss']:.6f}); "
            f"val_loss={val_metrics['loss']:.6f}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            trainer.save_checkpoint(args.checkpoint_path, epoch, best_val_loss)
            print(f"Saved best validation checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
