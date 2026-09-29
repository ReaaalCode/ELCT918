"""Data loading, reproducibility, result saving, and plotting helpers."""
from __future__ import annotations

import csv
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import CIFAR_STATS, TRAIN_FRACTION


def set_seed(seed: int) -> None:
    """Fix random sources so a repeated run starts from the same random state."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU: CUDA is not available.")
    return device


def default_num_workers() -> int:
    """Use a small safe worker count; Windows defaults to zero to avoid spawn issues."""
    if platform.system() == "Windows":
        return 0
    return min(4, os.cpu_count() or 1)


def make_dataloaders(
    dataset_name: str,
    data_dir: Path,
    batch_size: int,
    seed: int,
    num_workers: int,
    device: torch.device,
    train_subset: int | None = None,
    val_subset: int | None = None,
    test_subset: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create fixed train/validation split and preserve the official test set."""
    try:
        from torchvision import datasets, transforms
    except ImportError as error:
        raise ImportError("torchvision is required. Install the packages in requirements.txt.") from error

    mean, std = CIFAR_STATS[dataset_name]
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    dataset_class = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100

    # Two views of the same official training data let training use augmentation
    # while validation uses only evaluation preprocessing. The test set is never
    # included in this split.
    train_source = dataset_class(root=data_dir, train=True, download=True, transform=train_transform)
    val_source = dataset_class(root=data_dir, train=True, download=True, transform=eval_transform)
    test_source = dataset_class(root=data_dir, train=False, download=True, transform=eval_transform)

    indices = torch.randperm(len(train_source), generator=torch.Generator().manual_seed(seed)).tolist()
    train_size = int(len(indices) * TRAIN_FRACTION)
    train_indices, val_indices = indices[:train_size], indices[train_size:]
    if train_subset is not None:
        train_indices = train_indices[:train_subset]
    if val_subset is not None:
        val_indices = val_indices[:val_subset]
    test_indices = list(range(len(test_source)))
    if test_subset is not None:
        test_indices = test_indices[:test_subset]

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(Subset(train_source, train_indices), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(val_source, val_indices), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(Subset(test_source, test_indices), shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def ensure_directories(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_history(history: list[dict[str, Any]], path: Path) -> None:
    """Save the per-epoch metrics as both CSV and JSON next to the final results."""
    csv_path = path.with_suffix(".csv")
    json_path = path.with_suffix(".json")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def save_plots(history: list[dict[str, Any]], model_name: str, dataset_name: str, plots_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    title_prefix = f"{model_name.upper()} on {dataset_name.upper()}"

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="Training accuracy")
    plt.plot(epochs, [row["validation_accuracy"] for row in history], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{title_prefix}: accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{model_name}_{dataset_name}_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Training loss")
    plt.plot(epochs, [row["validation_loss"] for row in history], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"{title_prefix}: loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{model_name}_{dataset_name}_loss.png", dpi=160)
    plt.close()


SUMMARY_FIELDS = [
    "architecture",
    "dataset",
    "trainable_parameters",
    "test_top1_accuracy",
    "test_top5_accuracy",
    "accuracy_drop_percent",
    "average_epoch_time_seconds",
]


def update_summary(summary_path: Path, row: dict[str, Any]) -> None:
    """Replace one architecture/dataset row so reruns keep one clean final table."""
    rows: list[dict[str, str]] = []
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    row_as_text = {field: row.get(field, "") for field in SUMMARY_FIELDS}
    rows = [
        old_row
        for old_row in rows
        if not (old_row["architecture"] == row_as_text["architecture"] and old_row["dataset"] == row_as_text["dataset"])
    ]
    rows.append(row_as_text)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
