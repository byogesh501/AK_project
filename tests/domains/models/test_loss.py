"""Focused supervision tests for the PCB detector loss."""

import torch

from domains.pcb.models import PCBLoss


NUM_CLASSES = 3


def _target(class_id=0, width=0.125, height=0.125):
    # At 8x8, (0.5625, 0.5625) belongs to grid cell (4, 4), whose
    # zero-valued dx/dy prediction decodes to that same cell center.
    return torch.tensor([[[class_id, 0.5625, 0.5625, width, height]]], dtype=torch.float32)


def _predictions():
    """Three maps for a 64x64 synthetic image, with a match at stride 8."""
    predictions = [
        torch.zeros(1, 5 + NUM_CLASSES, 8, 8),
        torch.zeros(1, 5 + NUM_CLASSES, 4, 4),
        torch.zeros(1, 5 + NUM_CLASSES, 2, 2),
    ]
    # sigmoid(-1.94591) == 0.125, matching the default target dimensions.
    predictions[0][0, 2:4, 4, 4] = -1.94591
    return [prediction.requires_grad_() for prediction in predictions]


def test_ground_truth_box_changes_box_loss():
    criterion = PCBLoss(num_classes=NUM_CLASSES)
    mask = torch.tensor([[True]])

    matching = criterion(_predictions(), mask, _target(width=0.125, height=0.125))
    different_box = criterion(_predictions(), mask, _target(width=0.25, height=0.25))

    assert matching["l_box"] < different_box["l_box"]


def test_ground_truth_class_changes_classification_loss():
    criterion = PCBLoss(num_classes=NUM_CLASSES)
    mask = torch.tensor([[True]])
    predictions = _predictions()
    predictions[0].data[0, 5, 4, 4] = 8
    predictions[0].data[0, 6, 4, 4] = -8

    class_zero = criterion(predictions, mask, _target(class_id=0))
    class_one = criterion(predictions, mask, _target(class_id=1))

    assert class_zero["l_cls"] < class_one["l_cls"]


def test_padding_mask_excludes_padded_annotations():
    criterion = PCBLoss(num_classes=NUM_CLASSES)
    targets_with_padding = torch.tensor(
        [[[0, 0.5625, 0.5625, 0.125, 0.125], [2, 0.2, 0.2, 0.9, 0.9]]],
        dtype=torch.float32,
    )
    targets_without_padding = torch.tensor(
        [[[0, 0.5625, 0.5625, 0.125, 0.125], [0, 0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, False]])
    masked = criterion(_predictions(), mask, targets_with_padding)
    without_padding = criterion(_predictions(), mask, targets_without_padding)

    assert torch.allclose(masked["loss"], without_padding["loss"])
    assert torch.allclose(masked["l_box"], without_padding["l_box"])
    assert torch.allclose(masked["l_cls"], without_padding["l_cls"])


def test_valid_ground_truth_creates_positive_objectness_supervision():
    criterion = PCBLoss(num_classes=NUM_CLASSES)
    positive_prediction = _predictions()
    positive_prediction[0].data[0, 4, 4, 4] = 8

    with_gt = criterion(positive_prediction, torch.tensor([[True]]), _target())
    same_prediction_without_gt = [
        prediction.detach().clone().requires_grad_() for prediction in positive_prediction
    ]
    without_gt = criterion(same_prediction_without_gt, torch.tensor([[False]]), _target())

    # A high objectness logit is rewarded only when a valid GT selects its cell.
    assert with_gt["l_obj"] < without_gt["l_obj"]


def test_loss_backpropagates_through_prediction_maps():
    criterion = PCBLoss(num_classes=NUM_CLASSES)
    predictions = _predictions()
    targets = torch.tensor(
        [[[0, 0.5625, 0.5625, 0.125, 0.125], [1, 0.1875, 0.1875, 0.125, 0.125]]],
        dtype=torch.float32,
    )

    loss = criterion(predictions, torch.tensor([[True, True]]), targets)["loss"]
    loss.backward()

    assert all(prediction.grad is not None for prediction in predictions)
    assert predictions[0].grad.abs().sum() > 0
