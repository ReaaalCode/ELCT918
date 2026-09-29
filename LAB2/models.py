"""Custom CIFAR-sized LeNet-5, AlexNet, and VGG16 implementations."""
from __future__ import annotations

import torch
from torch import nn


class LeNet5(nn.Module):
    """LeNet-5 adapted only for RGB CIFAR images and a configurable output size."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # CIFAR is RGB, so the original one-channel input becomes three channels.
        # With a 32x32 input, the original 5x5 valid convolutions and 2x2 average
        # pools still produce a 16x5x5 map. C5 is kept as a 5x5 convolution, as
        # in LeNet-5; C3 uses full connectivity as a simple PyTorch simplification.
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.Tanh(),
            # Only this final layer changes between CIFAR-10 and CIFAR-100.
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class AlexNet(nn.Module):
    """AlexNet-style network for 32x32 CIFAR images."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # Original AlexNet begins with 11x11, stride 4 for 224x224 images. For a
        # 32x32 CIFAR image that would discard too much spatial information, so a
        # 5x5, stride-1 first filter keeps AlexNet's larger early receptive field.
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # CIFAR's 32x32 input becomes 3x3 after AlexNet's three overlapping pools.
        # The original two 4096-wide FC layers are retained; only their input size
        # changes because the CIFAR feature map is 256x3x3 rather than 256x6x6.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _vgg_block(in_channels: int, out_channels: int, conv_count: int) -> list[nn.Module]:
    layers: list[nn.Module] = []
    for index in range(conv_count):
        layers.extend(
            [
                nn.Conv2d(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
            ]
        )
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return layers


class VGG16(nn.Module):
    """VGG16 with its original five convolution blocks and CIFAR classifier."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # The 13 convolution layers retain VGG16's 3x3-only design and five pools.
        # Five pools map CIFAR's 32x32 input to 1x1, so no spatial redesign or
        # BatchNorm is added.
        self.features = nn.Sequential(
            *_vgg_block(3, 64, 2),
            *_vgg_block(64, 128, 2),
            *_vgg_block(128, 256, 3),
            *_vgg_block(256, 512, 3),
            *_vgg_block(512, 512, 3),
        )
        # Original VGG16 uses 4096-wide FC layers after a 7x7 feature map. CIFAR
        # reaches 1x1, so only the first FC input becomes 512; the two hidden
        # classifier widths stay at the original 4096 units.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Use the initialization needed for stable deep ReLU training without BatchNorm."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(name: str, num_classes: int) -> nn.Module:
    """Create one of the three custom models without pretrained weights."""
    models = {"lenet": LeNet5, "alexnet": AlexNet, "vgg16": VGG16}
    try:
        return models[name.lower()](num_classes)
    except KeyError as error:
        raise ValueError(f"Unknown model: {name}. Choose from {sorted(models)}.") from error
