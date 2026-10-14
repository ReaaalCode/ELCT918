"""LeNet-5 used for ELCT918 Lab 3 MNIST recognition."""
from __future__ import annotations

import torch
from torch import nn


class LeNet5MNIST(nn.Module):
    """Lab 2 LeNet-5 structure adapted only from RGB to MNIST grayscale input."""

    def __init__(self) -> None:
        super().__init__()
        # MNIST is padded from 28x28 to 32x32 before it enters this layer.
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # This is the Lab 2 C5-style layer; a 5x5 map becomes 1x1.
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
