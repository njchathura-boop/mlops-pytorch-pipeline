"""Small image-classification models for CIFAR-10."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class SimpleCNN(nn.Module):
    """A lightweight CNN for 32 x 32 RGB images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        features = torch.flatten(features, start_dim=1)
        return self.classifier(features)


def get_model(architecture: str, num_classes: int) -> nn.Module:
    """Create a supported model by name."""

    name = architecture.strip().lower()

    if name in {"simple_cnn", "cnn"}:
        return SimpleCNN(num_classes=num_classes)

    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported architecture: {architecture}. "
        "Use 'simple_cnn' or 'resnet18'."
    )
