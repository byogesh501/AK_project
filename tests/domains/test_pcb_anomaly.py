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
    DeepPCBTemplateAnomalyDataset,
    MODEL_INPUT_CHANNELS,
    SyntheticNormalTransform,
    apply_threshold,
    anomaly_collate_fn,
    build_pair_difference_input,
    calibrate_validation_threshold,
    reconstruction_error_map,
    score_model_inputs,
    score_labeled_batch,
    score_pair_inputs,
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
    model_input = torch.rand(2, MODEL_INPUT_CHANNELS, 64, 64)

    reconstruction = model(model_input)

    assert reconstruction.shape == (2, 3, 64, 64)
    assert torch.all(reconstruction >= 0)
    assert torch.all(reconstruction <= 1)


def test_autoencoder_rejects_legacy_pair_input():
    model = ConvAutoencoder(base_channels=4)
    with pytest.raises(ValueError, match="3 channels"):
        model(torch.rand(1, 9, 32, 32))


def test_autoencoder_training_step_backpropagates():
    model = ConvAutoencoder(base_channels=4)
    trainer = AnomalyTrainer(model, AnomalyTrainingConfig(device="cpu"))
    batch = {
        "pair_inputs": torch.rand(2, MODEL_INPUT_CHANNELS, 32, 32),
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
    assert batch["annotations"] == [sample["annotations"]]


def test_anomaly_dataset_keeps_test_split_isolated(tmp_path):
    with pytest.raises(ValueError, match="test split is isolated"):
        DeepPCBAnomalyDataset(str(tmp_path), str(tmp_path / "manifest.csv"), "test")


def test_synthetic_normal_transform_is_deterministic_and_non_identity():
    template = torch.full((3, 16, 16), 0.5)
    transform = SyntheticNormalTransform(seed=17)

    first = transform(template, 3)
    second = transform(template, 3)

    assert torch.equal(first, second)
    assert not torch.equal(first, template)
    assert torch.all((first >= 0) & (first <= 1))


def test_template_dataset_uses_only_three_channel_synthetic_input(tmp_path):
    image_dir = tmp_path / "group1" / "1"
    annotation_dir = tmp_path / "group1" / "1_not"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    Image.new("RGB", (16, 16), (255, 0, 0)).save(image_dir / "0001_test.jpg")
    Image.new("RGB", (16, 16), (0, 255, 0)).save(image_dir / "0001_temp.jpg")
    (annotation_dir / "0001.txt").write_text("0 0 4 4 1\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["split", "pair_id", "group", "test_img", "temp_img", "anno_file", "classes"],
        )
        writer.writeheader()
        for split in ("train", "val"):
            writer.writerow(
                {
                    "split": split,
                    "pair_id": f"group1_{split}",
                    "group": "group1",
                    "test_img": "group1/1/0001_test.jpg",
                    "temp_img": "group1/1/0001_temp.jpg",
                    "anno_file": "group1/1_not/0001.txt",
                    "classes": "open_circuit",
                }
            )

    train = DeepPCBTemplateAnomalyDataset(str(tmp_path), str(manifest_path), "train", variation_seed=1)
    val = DeepPCBTemplateAnomalyDataset(str(tmp_path), str(manifest_path), "val", variation_seed=2)
    train_sample = train[0]
    val_sample = val[0]

    assert train_sample["pair_input"].shape == (3, 16, 16)
    assert train_sample["normal_target"].shape == (3, 16, 16)
    assert not torch.equal(train_sample["pair_input"], train_sample["normal_target"])
    assert set(train.pair_ids).isdisjoint(val.pair_ids)
    assert train_sample["pair_id"] != val_sample["pair_id"]


def test_template_dataset_rejects_test_split(tmp_path):
    with pytest.raises(ValueError, match="test split is isolated"):
        DeepPCBTemplateAnomalyDataset(str(tmp_path), str(tmp_path / "manifest.csv"), "test")


class _ZeroReconstruction(torch.nn.Module):
    def forward(self, pair_inputs):
        return torch.zeros_like(pair_inputs[:, :3])


def test_scores_use_inspected_image_and_return_spatial_error_map():
    pair_inputs = torch.zeros(2, 9, 16, 16)
    pair_inputs[0, :3] = 1.0

    result = score_pair_inputs(_ZeroReconstruction(), pair_inputs)

    assert result["error_maps"].shape == (2, 16, 16)
    assert torch.all((result["error_maps"] >= 0) & (result["error_maps"] <= 1))
    assert torch.equal(result["error_maps"][0], torch.ones(16, 16))
    assert torch.equal(result["image_scores"], torch.tensor([1.0, 0.0]))


def test_model_scoring_accepts_three_channel_inputs():
    result = score_model_inputs(_ZeroReconstruction(), torch.zeros(2, 3, 16, 16))

    assert result["reconstructions"].shape == (2, 3, 16, 16)
    assert result["error_maps"].shape == (2, 16, 16)


def test_reconstruction_error_map_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        reconstruction_error_map(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 7, 8))


def test_validation_threshold_calibration_and_application():
    scores = torch.tensor([0.1, 0.2, 0.3, 0.9])
    calibration = calibrate_validation_threshold(
        scores, torch.tensor([True, True, True, False]), quantile=0.5
    )

    assert calibration.threshold == pytest.approx(0.2)
    assert calibration.normal_count == 3
    assert torch.equal(apply_threshold(scores, calibration.threshold), torch.tensor([False, True, True, True]))


def test_threshold_calibration_requires_validation_and_normal_labels():
    scores = torch.tensor([0.1, 0.2])
    unavailable = calibrate_validation_threshold(scores, torch.tensor([False, False]))

    assert unavailable.threshold is None
    assert unavailable.limitation is not None
    with pytest.raises(ValueError, match="validation split"):
        calibrate_validation_threshold(scores, torch.tensor([True, True]), split="test")


def test_labeled_batch_keeps_known_annotations_separate_from_scores():
    batch = {
        "pair_inputs": torch.zeros(1, 9, 16, 16),
        "pair_ids": ["pair-1"],
        "annotations": [[{"class_id": 0}]],
    }

    result = score_labeled_batch(_ZeroReconstruction(), batch)

    assert result["pair_ids"] == ["pair-1"]
    assert result["annotations"] == batch["annotations"]
    assert result["image_scores"].shape == (1,)
