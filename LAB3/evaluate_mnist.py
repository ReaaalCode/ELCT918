"""Evaluate the saved Lab 3 MNIST LeNet state_dict without training."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from config import BATCH_SIZE, DATA_DIR, WEIGHTS_PATH
from models import LeNet5MNIST, count_trainable_parameters
from train_mnist import evaluate, mnist_transform


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}. Run train_mnist.py first.")
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=mnist_transform())
    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    model = LeNet5MNIST().to(device)
    try:
        state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    loss, accuracy = evaluate(model, loader, device)
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"MNIST test loss: {loss:.4f}")
    print(f"MNIST test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
