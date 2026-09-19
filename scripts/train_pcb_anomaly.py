"""Train the isolated PCB anomaly autoencoder from scratch on DeepPCB pairs."""

import argparse
from copy import deepcopy
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from domains.pcb.anomaly import (
    AnomalyTrainer,
    AnomalyTrainingConfig,
    DeepPCBTemplateAnomalyDataset,
    ConvAutoencoder,
    anomaly_collate_fn,
    calibrate_validation_threshold,
    score_model_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-calibration-batches", type=int, default=None)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "pcb_anomaly_autoencoder_best.pt",
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
    dataset_args = {
        "data_root": str(data_root),
        "manifest_path": str(processed_root / "manifest.csv"),
        "class_map_path": str(processed_root / "class_map.json"),
    }
    # Train and calibration use templates only; test remains isolated.
    train_dataset = DeepPCBTemplateAnomalyDataset(split="train", variation_seed=args.seed, **dataset_args)
    validation_templates = DeepPCBTemplateAnomalyDataset(
        split="val", variation_seed=args.seed + 1, **dataset_args
    )
    calibration_ids = validation_templates.pair_ids[: len(validation_templates) // 2]
    validation_ids = validation_templates.pair_ids[len(validation_templates) // 2 :]
    val_dataset = DeepPCBTemplateAnomalyDataset(
        split="val",
        pair_ids=validation_ids,
        variation_seed=args.seed + 1,
        **dataset_args,
    )
    calibration_dataset = DeepPCBTemplateAnomalyDataset(
        split="val",
        pair_ids=calibration_ids,
        variation_seed=args.seed + 2,
        **dataset_args,
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device != "cpu" and torch.cuda.is_available(),
        "collate_fn": anomaly_collate_fn,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    calibration_loader = DataLoader(calibration_dataset, shuffle=False, **loader_args)

    model = ConvAutoencoder(base_channels=args.base_channels)
    trainer = AnomalyTrainer(model, AnomalyTrainingConfig(args.learning_rate, args.device))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    best_val_loss = float("inf")
    best_state_dict = None

    print(f"Device: {trainer.device}")
    print(
        f"Train templates: {len(train_dataset)}; validation pairs: {len(val_dataset)}; "
        f"calibration pairs: {len(calibration_dataset)}; test pairs: unused"
    )
    print(f"Trainable parameters: {parameter_count:,}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(train_loader, args.max_train_batches)
        val_metrics = trainer.evaluate(val_loader, args.max_val_batches)
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_metrics['loss']:.6f}; val_loss={val_metrics['loss']:.6f}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state_dict = deepcopy(model.state_dict())
            trainer.save_checkpoint(args.checkpoint_path, epoch, best_val_loss)
            print(f"Saved best validation checkpoint: {args.checkpoint_path}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    calibration_scores = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(calibration_loader):
            if args.max_calibration_batches is not None and batch_index >= args.max_calibration_batches:
                break
            result = score_model_inputs(
                model, batch["pair_inputs"].to(trainer.device, non_blocking=True)
            )
            calibration_scores.append(result["image_scores"].cpu())
    if calibration_scores:
        scores = torch.cat(calibration_scores)
        calibration = calibrate_validation_threshold(
            scores, torch.ones_like(scores, dtype=torch.bool), split="val"
        )
        print(
            f"Synthetic-normal validation threshold: {calibration.threshold:.6f} "
            f"({calibration.normal_count} calibration pairs)"
        )


if __name__ == "__main__":
    main()
