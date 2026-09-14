"""
Minimal testing for the Phase 2A custom detector initialization,
forward pass, multi-scale structural checks, and backward hooks.
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from domains.pcb.models import PCBDetector, PCBLoss
from domains.pcb.config import NUM_CLASSES

def test_detector_initialization():
    """Verify the detector instantiates as expected (train from scratch policy)."""
    model = PCBDetector(in_channels=3, base_channels=16, num_classes=NUM_CLASSES)
    assert model is not None
    assert len(list(model.parameters())) > 0

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\\nDetector parameters: {param_count:,}")

def test_detector_forward_shapes():
    """Verify multi-scale feature maps at strides 8, 16, 32."""
    model = PCBDetector(in_channels=3, base_channels=16, num_classes=NUM_CLASSES)
    model.eval()

    # DeepPCB tensor shape: B, C, H, W (640x640)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 640, 640)

    with torch.no_grad():
        outputs = model(dummy_input)

    assert len(outputs) == 3, "Expected 3 multi-scale outputs"

    # 4 coords + 1 obj + 6 classes = 11 channels
    expected_channels = 4 + 1 + NUM_CLASSES

    s8, s16, s32 = outputs
    assert s8.shape == (batch_size, expected_channels, 640 // 8, 640 // 8) # 80x80
    assert s16.shape == (batch_size, expected_channels, 640 // 16, 640 // 16) # 40x40
    assert s32.shape == (batch_size, expected_channels, 640 // 32, 640 // 32) # 20x20


def test_detector_backward_pass():
    """Verify that gradients flow back to the input layers."""
    model = PCBDetector(in_channels=3, base_channels=16, num_classes=NUM_CLASSES)
    model.train()

    criterion = PCBLoss(num_classes=NUM_CLASSES)

    batch_size = 2
    # A small image keeps this test a fast synthetic gradient check; model
    # output dimensions are verified at DeepPCB's native 640x640 above.
    dummy_input = torch.randn(batch_size, 3, 64, 64, requires_grad=True)
    targets = torch.tensor([
        [[0, 0.5625, 0.5625, 0.125, 0.125], [1, 0.1875, 0.1875, 0.125, 0.125]],
        [[2, 0.6875, 0.3125, 0.125, 0.125], [0, 0.0, 0.0, 0.0, 0.0]],
    ])
    targets_mask = torch.tensor([[True, True], [True, False]])

    outputs = model(dummy_input)

    loss_dict = criterion(outputs, targets_mask, targets)
    loss = loss_dict['loss']

    loss.backward()

    # Verify backward pass was successful
    assert dummy_input.grad is not None, "Gradients should flow to input"

    # Verify first layer weights received gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad, "Model parameters should receive gradients"
