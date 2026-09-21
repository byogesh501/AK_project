"""Evaluate the frozen ConvAE on DeepPCB test pairs (known defects only)."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from domains.pcb.anomaly import (
    AnomalyPairTransform,
    ConvAutoencoder,
    build_pair_difference_input,
    evaluate_known_defect_batches,
    per_class_known_defect_metrics,
    score_statistics,
)
from domains.pcb.preprocessing.dataset import DeepPCBDataset
from domains.pcb.training.trainer import resolve_device


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FROZEN_THRESHOLD = 0.003249


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "pcb_anomaly_autoencoder_best.pt",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None,
                        help="Force cpu or cuda; auto-detects if omitted.")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Cap batches for smoke testing; default evaluates all 225 test pairs.")
    return parser.parse_args()


def _load_model(checkpoint_path: Path, device: torch.device) -> ConvAutoencoder:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Anomaly checkpoint not found: {checkpoint_path}. "
            "Supply the frozen pcb_anomaly_autoencoder_best.pt; no model is trained here."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    first_weight = state_dict.get("encoder.0.0.weight")
    if first_weight is None:
        raise ValueError("Checkpoint does not contain a ConvAutoencoder state_dict")
    model = ConvAutoencoder(base_channels=int(first_weight.shape[0]))
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _test_collate_fn(batch):
    """Collate DeepPCBDataset test samples into the evaluation batch format.

    DeepPCBDataset returns defect_image/template_image keys; the evaluation
    loop expects pair_inputs (9-ch) and annotations.
    """
    pair_inputs = torch.stack([
        build_pair_difference_input(s["defect_image"], s["template_image"])
        for s in batch
    ])
    return {
        "pair_inputs": pair_inputs,
        "pair_ids": [s["pair_id"] for s in batch],
        "annotations": [s["annotations"] for s in batch],
    }


def main():
    args = parse_args()
    device = resolve_device(args.device or "auto")

    data_root = PROJECT_ROOT / "data" / "raw" / "deeppcb"
    processed_root = PROJECT_ROOT / "data" / "processed"
    test_dataset = DeepPCBDataset(
        data_root=str(data_root),
        manifest_path=str(processed_root / "manifest.csv"),
        split="test",
        class_map_path=str(processed_root / "class_map.json"),
        transform=AnomalyPairTransform(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_test_collate_fn,
    )

    model = _load_model(args.checkpoint_path, device)
    result = evaluate_known_defect_batches(
        model, test_loader, FROZEN_THRESHOLD, device=device, max_batches=args.max_batches,
    )

    test_scores = result["image_scores"]
    flagged = int(result["predictions"].sum())
    total = len(result["pair_ids"])

    summary = {
        "evaluation": (
            "KNOWN-DEFECT DeepPCB test evaluation. These are descriptive "
            "known-defect metrics, NOT unknown-anomaly classification metrics."
        ),
        "split": "test",
        "pair_count": total,
        "threshold": FROZEN_THRESHOLD,
        "threshold_source": "Frozen from 112 synthetic-normal validation calibration pairs",
        "score_statistics": score_statistics(test_scores),
        "above_threshold_count": flagged,
        "above_threshold_pct": round(100.0 * flagged / total, 2) if total else 0.0,
        "image_level": result["metrics"],
        "localization": result["localization"],
        "per_class": per_class_known_defect_metrics(
            test_scores,
            result["error_maps"],
            result["annotations"],
            FROZEN_THRESHOLD,
            test_dataset.class_map,
        ),
        "false_positive_limitation": (
            "DeepPCB contains no normal (defect-free) test images. "
            "False-positive rate, specificity, and ROC-AUC cannot be "
            "established from this dataset."
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
