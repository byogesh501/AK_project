"""Unit tests for the isolated PCB anomaly autoencoder foundation."""

import csv

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from domains.pcb.anomaly import (
    AnomalyTrainer,
    AnomalyTrainingConfig,
    ConvAutoencoder,
    DeepPCBAnomalyDataset,
    anomaly_collate_fn,
    build_pair_difference_input,
)


def test_pair_difference_input_preserves_pair_and_signed_difference():
    defect = torch.full((3, 8, 8), 0.75)
    template = torch.full((3, 8, 8), 0.25)

    pair_input = build_pair_difference_input(defect, template)

    assert pair_input.shape == (9, 8, 8)
    assert torch.equal(pair_input[:3], defect)
    assert torch.equal(pair_input[3:6], template)
    assert torch.equal(pair_input[6:], defect - template)


def test_pair_difference_input_rejects_unaligned_tensors():
    with pytest.raises(ValueError, match="identical shapes"):
        build_pair_difference_input(torch.rand(3, 8, 8), torch.rand(3, 7, 8))


def test_autoencoder_forward_shape_and_range():
    model = ConvAutoencoder(base_channels=8)
    pair_input = torch.rand(2, 9, 64, 64)

    reconstruction = model(pair_input)

    assert reconstruction.shape == (2, 3, 64, 64)
    assert torch.all(reconstruction >= 0)
    assert torch.all(reconstruction <= 1)


def test_autoencoder_training_step_backpropagates():
    model = ConvAutoencoder(base_channels=4)
    trainer = AnomalyTrainer(model, AnomalyTrainingConfig(device="cpu"))
    batch = {
        "pair_inputs": torch.rand(2, 9, 32, 32),
        "normal_targets": torch.rand(2, 3, 32, 32),
    }
    before = next(model.parameters()).detach().clone()

    metrics = trainer.train_epoch([batch])

    assert metrics["batches"] == 1
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(before, next(model.parameters()).detach())


def test_anomaly_dataset_uses_template_as_normal_target(tmp_path):
    image_dir = tmp_path / "group1" / "1"
    annotation_dir = tmp_path / "group1" / "1_not"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    Image.new("RGB", (16, 16), (255, 0, 0)).save(image_dir / "0001_test.jpg")
    Image.new("RGB", (16, 16), (0, 255, 0)).save(image_dir / "0001_temp.jpg")
    # DeepPCB annotation parser expects x1, y1, x2, y2, defect_type.
    (annotation_dir / "0001.txt").write_text("0 0 4 4 1\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["split", "pair_id", "group", "test_img", "temp_img", "anno_file", "classes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "pair_id": "group1_0001",
                "group": "group1",
                "test_img": "group1/1/0001_test.jpg",
                "temp_img": "group1/1/0001_temp.jpg",
                "anno_file": "group1/1_not/0001.txt",
                "classes": "open_circuit",
            }
        )

    dataset = DeepPCBAnomalyDataset(str(tmp_path), str(manifest_path), "train")
    sample = dataset[0]
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=anomaly_collate_fn)))

    assert sample["pair_input"].shape == (9, 16, 16)
    assert torch.equal(sample["normal_target"], sample["pair_input"][3:6])
    assert sample["normal_target"].mean() != sample["pair_input"][:3].mean()
    assert batch["pair_inputs"].shape == (1, 9, 16, 16)
    assert batch["normal_targets"].shape == (1, 3, 16, 16)


def test_anomaly_dataset_keeps_test_split_isolated(tmp_path):
    with pytest.raises(ValueError, match="test split is isolated"):
        DeepPCBAnomalyDataset(str(tmp_path), str(tmp_path / "manifest.csv"), "test")
