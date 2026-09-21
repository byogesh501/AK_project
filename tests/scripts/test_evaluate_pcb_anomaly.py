"""Focused tests for the evaluate_pcb_anomaly CLI helpers."""

import torch
import pytest

from scripts.evaluate_pcb_anomaly import _test_collate_fn, FROZEN_THRESHOLD


def test_frozen_threshold_matches_calibration():
    assert FROZEN_THRESHOLD == pytest.approx(0.003249)


def test_test_collate_fn_builds_pair_inputs_from_dataset_keys():
    defect = torch.rand(3, 16, 16)
    template = torch.rand(3, 16, 16)
    batch = [
        {
            "defect_image": defect,
            "template_image": template,
            "annotations": [{"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}],
            "pair_id": "test-1",
            "group": "g1",
            "classes": {"open_circuit"},
        },
    ]

    collated = _test_collate_fn(batch)

    assert collated["pair_inputs"].shape == (1, 9, 16, 16)
    assert torch.equal(collated["pair_inputs"][0, :3], defect)
    assert torch.equal(collated["pair_inputs"][0, 3:6], template)
    assert collated["pair_ids"] == ["test-1"]
    assert len(collated["annotations"]) == 1
