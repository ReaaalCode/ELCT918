"""Training and evaluation loops shared by every Lab 2 architecture."""
from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Train once and time only forward/backward/optimizer computation per batch."""
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    compute_seconds = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        _synchronize_if_cuda(device)
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became NaN or infinity.")
        loss.backward()
        optimizer.step()

        _synchronize_if_cuda(device)
        compute_seconds += time.perf_counter() - start
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return {
        "loss": loss_sum / total,
        "accuracy": 100.0 * correct / total,
        "compute_seconds": compute_seconds,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    compute_top5: bool = False,
) -> dict[str, Any]:
    """Evaluate without gradients and return correctly sample-weighted metrics."""
    model.eval()
    loss_sum = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        logits = model(images)
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Evaluation loss became NaN or infinity.")

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        top1_correct += (logits.argmax(dim=1) == labels).sum().item()
        if compute_top5:
            if logits.size(1) < 5:
                raise ValueError("Top-5 accuracy requires at least five output classes.")
            top5_predictions = logits.topk(k=5, dim=1).indices
            top5_correct += top5_predictions.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
        total += batch_size

    results: dict[str, Any] = {
        "loss": loss_sum / total,
        "top1_accuracy": 100.0 * top1_correct / total,
    }
    if compute_top5:
        results["top5_accuracy"] = 100.0 * top5_correct / total
    return results
