"""Train the Lab 3 LeNet-5 MNIST model from scratch."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import (BATCH_SIZE, DATA_DIR, EPOCHS, LEARNING_RATE, MNIST_MEAN,
                    MNIST_STD, MOMENTUM, RESULTS_DIR, SCHEDULER_GAMMA,
                    SCHEDULER_MILESTONES, SEED, WEIGHT_DECAY, WEIGHTS_PATH)
from models import LeNet5MNIST, count_trainable_parameters


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mnist_transform() -> transforms.Compose:
    # Padding happens in pixel space before normalization, matching webcam preprocessing.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2, fill=0),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Lab 3 LeNet-5 on MNIST.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs and batch-size must be positive; lr must be greater than zero.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    transform = mnist_transform()
    train_source = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    # Use a reproducible 55,000/5,000 split and keep the official test set untouched.
    indices = torch.randperm(len(train_source), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_set = Subset(train_source, indices[:55000])
    validation_set = Subset(train_source, indices[55000:])
    workers = min(4, os.cpu_count() or 1)
    loader_options = {"num_workers": workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, **loader_options)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **loader_options)

    model = LeNet5MNIST().to(device)
    parameter_count = count_trainable_parameters(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[m for m in SCHEDULER_MILESTONES if m < args.epochs], gamma=SCHEDULER_GAMMA)
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = correct = total = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became non-finite.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - start
        validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "training_loss": total_loss / total, "training_accuracy": 100.0 * correct / total, "validation_loss": validation_loss, "validation_accuracy": validation_accuracy, "training_epoch_seconds": epoch_seconds}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs}: train loss {row['training_loss']:.4f}, train accuracy {row['training_accuracy']:.2f}%, validation loss {validation_loss:.4f}, validation accuracy {validation_accuracy:.2f}%, time {epoch_seconds:.2f}s")
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), WEIGHTS_PATH)
        scheduler.step()

    # The saved state_dict must load into a new identical model before reporting success.
    fresh_model = LeNet5MNIST().to(device)
    try:
        state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    fresh_model.load_state_dict(state_dict)
    final_test_loss, final_test_accuracy = evaluate(fresh_model, test_loader, device)

    with (RESULTS_DIR / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metrics = {
        "test_accuracy_percent": final_test_accuracy,
        "test_loss": final_test_loss,
        "best_validation_accuracy_percent": best_accuracy,
        "trainable_parameters": parameter_count,
        "average_training_epoch_seconds": sum(row["training_epoch_seconds"] for row in history) / len(history),
        "individual_epoch_seconds": [row["training_epoch_seconds"] for row in history],
        "training_settings": {"seed": args.seed, "epochs": args.epochs, "optimizer": "SGD", "learning_rate": args.lr, "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY, "batch_size": args.batch_size, "scheduler": {"name": "MultiStepLR", "milestones": [m for m in SCHEDULER_MILESTONES if m < args.epochs], "gamma": SCHEDULER_GAMMA}, "loss_function": "CrossEntropyLoss", "split": {"training": 55000, "validation": 5000, "official_test": 10000}},
        "normalization": {"mean": MNIST_MEAN, "std": MNIST_STD, "input_pipeline": "ToTensor -> zero-pad 28x28 to 32x32 -> Normalize"},
        "device": str(device),
        "weights_path": str(WEIGHTS_PATH.name),
        "weights_reload_verified": True,
    }
    (RESULTS_DIR / "task1_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved state_dict: {WEIGHTS_PATH}")
    print(f"Reloaded test accuracy: {final_test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
