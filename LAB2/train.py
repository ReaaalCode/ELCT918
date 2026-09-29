"""One reusable training command for every Lab 2 model/dataset combination."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from architecture_report import write_architecture_report
from config import BATCH_SIZE, CHECKPOINTS_DIR, DATA_DIR, EPOCHS, LEARNING_RATE, MOMENTUM, NUM_CLASSES, PLOTS_DIR, RESULTS_DIR, SEED, WEIGHT_DECAY
from evaluate import evaluate, train_one_epoch
from models import build_model
from utils import count_trainable_parameters, default_num_workers, ensure_directories, get_device, make_dataloaders, save_history, save_plots, set_seed, update_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one custom CNN from scratch for ELCT918 Lab 2.")
    parser.add_argument("--model", choices=["lenet", "alexnet", "vgg16"], required=True)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=default_num_workers())
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--train-subset", type=int, default=None, help="Optional small training subset for debugging.")
    parser.add_argument("--val-subset", type=int, default=None, help="Optional small validation subset for debugging.")
    parser.add_argument("--test-subset", type=int, default=None, help="Optional small test subset for debugging.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one epoch on small subsets.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.num_workers < 0:
        parser.error("epochs, batch-size, and num-workers must be positive; lr must be greater than zero.")
    if args.smoke_test:
        args.epochs = 1
        args.train_subset = args.train_subset or 512
        args.val_subset = args.val_subset or 128
        args.test_subset = args.test_subset or 128
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    ensure_directories(RESULTS_DIR, PLOTS_DIR, CHECKPOINTS_DIR, args.data_dir)
    num_classes = NUM_CLASSES[args.dataset]
    model = build_model(args.model, num_classes).to(device)
    parameter_count = count_trainable_parameters(model)
    print(f"Model: {args.model}; dataset: {args.dataset}; trainable parameters: {parameter_count:,}")
    train_loader, val_loader, test_loader = make_dataloaders(args.dataset, args.data_dir, args.batch_size, args.seed, args.num_workers, device, args.train_subset, args.val_subset, args.test_subset)
    print(f"Samples - train: {len(train_loader.dataset)}, validation: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}")
    architecture_path = write_architecture_report(args.model, num_classes, RESULTS_DIR)
    print(f"Architecture summary: {architecture_path.name}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # Fixed policy: at the default 20 epochs, reduce LR after 10 and 15 epochs.
    milestones = sorted({max(1, args.epochs // 2), max(1, (3 * args.epochs) // 4)})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    history: list[dict[str, float]] = []
    best_validation_accuracy = float("-inf")
    checkpoint_path = CHECKPOINTS_DIR / f"{args.model}_{args.dataset}_best.pt"
    for epoch in range(1, args.epochs + 1):
        learning_rate_used = optimizer.param_groups[0]["lr"]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_epoch_seconds = time.perf_counter() - epoch_start
        validation_metrics = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        row = {"epoch": epoch, "learning_rate": learning_rate_used, "train_loss": train_metrics["loss"], "validation_loss": validation_metrics["loss"], "train_accuracy": train_metrics["accuracy"], "validation_accuracy": validation_metrics["top1_accuracy"], "training_epoch_seconds": training_epoch_seconds, "training_compute_seconds": train_metrics["compute_seconds"]}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs}: train loss {row['train_loss']:.4f}, train accuracy {row['train_accuracy']:.2f}%, validation loss {row['validation_loss']:.4f}, validation accuracy {row['validation_accuracy']:.2f}%, epoch time {row['training_epoch_seconds']:.2f}s")
        if row["validation_accuracy"] > best_validation_accuracy:
            best_validation_accuracy = row["validation_accuracy"]
            torch.save(model.state_dict(), checkpoint_path)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # Supports older PyTorch versions without weights_only.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    test_metrics = evaluate(model, test_loader, loss_fn, device, compute_top5=args.dataset == "cifar100")
    average_epoch_time = sum(row["training_epoch_seconds"] for row in history) / len(history)
    final_metrics = {"architecture": args.model, "dataset": args.dataset, "trainable_parameters": parameter_count, "best_validation_accuracy": best_validation_accuracy, "test_top1_accuracy": test_metrics["top1_accuracy"], "test_top5_accuracy": test_metrics.get("top5_accuracy"), "accuracy_drop_percent": 100.0 - test_metrics["top1_accuracy"], "average_epoch_time_seconds": average_epoch_time, "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr, "optimizer": "SGD", "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY, "seed": args.seed, "device": str(device)}
    result_stem = RESULTS_DIR / f"{args.model}_{args.dataset}_metrics"
    save_history(history, result_stem)
    (RESULTS_DIR / f"{args.model}_{args.dataset}_final.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    save_plots(history, args.model, args.dataset, PLOTS_DIR)
    update_summary(RESULTS_DIR / "summary.csv", final_metrics)
    print(f"Test top-1 accuracy: {final_metrics['test_top1_accuracy']:.2f}%")
    if final_metrics["test_top5_accuracy"] is not None:
        print(f"Test top-5 accuracy: {final_metrics['test_top5_accuracy']:.2f}%")
    print(f"Accuracy drop: {final_metrics['accuracy_drop_percent']:.2f}%")
    print(f"Saved metrics to {RESULTS_DIR}, plots to {PLOTS_DIR}, and checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
