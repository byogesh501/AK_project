#!/usr/bin/env python3
"""Minimal live demo for the trained PCB defect detector."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image, ImageDraw, ImageFont

from domains.pcb.config import CLASS_NAMES
from domains.pcb.models.detector import PCBDetector
from domains.pcb.preprocessing.transforms import PairedNormalize, PairedToTensor


DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "pcb_detector_best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "demo_detection.jpg"


def preprocess_image(image_path: Path, size: tuple[int, int] = (640, 640)):
    """Load an RGB image and convert it into the detector's expected normalized tensor."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    resized = image.resize(size, Image.Resampling.BILINEAR)

    to_tensor = PairedToTensor()
    tensor, _ = to_tensor(resized, resized)

    normalize = PairedNormalize()
    tensor, _ = normalize(tensor, tensor)

    return tensor.unsqueeze(0), image, original_size


def draw_prediction(draw, bbox, label, color):
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    text = label
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    box = [x1, max(0, y1 - text_h - 6), x1 + text_w + 8, max(0, y1)]
    draw.rectangle(box, fill=color)
    draw.text((x1 + 4, max(0, y1 - text_h - 4)), text, fill="white", font=font)


def main():
    parser = argparse.ArgumentParser(description="Run PCB defect detection on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to a single RGB PCB image.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the trained detector checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Where to save the annotated output image.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PCBDetector(in_channels=3, num_classes=6).to(device)
    model.eval()

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            torch.save(checkpoint["model_state_dict"], tmp_file.name)
            tmp_path = tmp_file.name
        try:
            model.load(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        model.load(str(checkpoint_path))

    x, source_image, original_size = preprocess_image(image_path)
    x = x.to(device)

    with torch.no_grad():
        detections = model.predict(x, conf_threshold=0.25, nms_iou_threshold=0.5, max_detections=300)[0]

    annotated = source_image.copy()
    draw = ImageDraw.Draw(annotated)
    out_w, out_h = original_size

    if detections.numel() == 0:
        print(f"No detections found for {image_path}.")
    else:
        detections = detections.cpu()
        for index, det in enumerate(detections, 1):
            x1n, y1n, x2n, y2n, confidence, class_id = det.tolist()
            class_id = int(class_id)
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

            x1 = x1n * out_w
            y1 = y1n * out_h
            x2 = x2n * out_w
            y2 = y2n * out_h

            x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
            x1 = max(0, min(out_w, x1))
            y1 = max(0, min(out_h, y1))
            x2 = max(0, min(out_w, x2))
            y2 = max(0, min(out_h, y2))

            color = ((index * 53) % 256, (index * 97) % 256, (index * 167) % 256)
            label = f"{class_name}: {confidence:.2f}"
            draw_prediction(draw, (x1, y1, x2, y2), label, color)

            print(
                f"[{index}] class={class_name}, confidence={confidence:.3f}, "
                f"bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    print(f"Saved annotated image to: {output_path}")


if __name__ == "__main__":
    main()
