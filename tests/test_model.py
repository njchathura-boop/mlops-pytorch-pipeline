"""Unit tests for the model factory."""

import pytest
import torch

from src.model import get_model


def test_simple_cnn_output_shape() -> None:
    model = get_model("simple_cnn", num_classes=10)
    inputs = torch.randn(4, 3, 32, 32)
    outputs = model(inputs)
    assert outputs.shape == (4, 10)


def test_resnet18_output_shape() -> None:
    model = get_model("resnet18", num_classes=10)
    inputs = torch.randn(2, 3, 32, 32)
    outputs = model(inputs)
    assert outputs.shape == (2, 10)


def test_unknown_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported architecture"):
        get_model("unknown", num_classes=10)
