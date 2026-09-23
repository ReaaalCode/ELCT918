#!/usr/bin/env python3
"""ELCT918 Lab 1: exhaustive MNIST MLP design-space exploration with PyTorch."""
from __future__ import annotations # Better annotations

import argparse
import csv
import gzip
import random
import struct
import time
import urllib.request # Downloads MNIST from the internet
from pathlib import Path

import matplotlib.pyplot as plt # Creates the Pareto-front plot
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 918 # Fixed random seed to make sure the random sequence is consistent between tests
MNIST_URLS = {  # URLs of datasets
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def set_seed(seed: int = SEED) -> None: # Gets random sequence from seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_mnist(data_dir: Path) -> dict[str, Path]: # Downloading MNIST into data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in MNIST_URLS.items():
        path = data_dir / url.rsplit("/", 1)[-1]
        paths[name] = path
        if not path.exists():
            print(f"Downloading {path.name} ...")
            urllib.request.urlretrieve(url, path)
    return paths


def read_idx(path: Path) -> np.ndarray: # Reads the 16 header bytes which describe the magic number, number of images, image height and image width
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4)) # Magic number identifies the files's format and data type, pretty much it says what kind of data to expect 
        ndim = magic & 0xFF
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(shape)


def load_mnist(data_dir: Path): # Created a 2D matrix for each dataset, where its (index of image, pixels of image), and pixels of image is the flattened 28x28 image so a total of 784 pixels organized left to right, up to down.
    p = download_mnist(data_dir)
    x_train = read_idx(p["train_images"]).reshape(-1, 784).astype(np.float32) / 255.0
    y_train = read_idx(p["train_labels"]).astype(np.int64)
    x_test = read_idx(p["test_images"]).reshape(-1, 784).astype(np.float32) / 255.0
    y_test = read_idx(p["test_labels"]).astype(np.int64)
    return x_train, y_train, x_test, y_test


def build_model(num_layers: int, nodes_per_layer: int) -> nn.Module: # Creates neural-network architecture with parameterizable number of players and layer width
    """Build 784 -> n hidden layers of m units -> 10 programmatically."""
    if num_layers < 1 or nodes_per_layer < 1:
        raise ValueError("num_layers and nodes_per_layer must be positive")
    layers: list[nn.Module] = []
    in_features = 784
    for _ in range(num_layers):
        layers.extend((nn.Linear(in_features, nodes_per_layer), nn.ReLU()))
        in_features = nodes_per_layer
    layers.append(nn.Linear(in_features, 10))
    return nn.Sequential(*layers)


def cost_model(num_layers: int, nodes_per_layer: int) -> tuple[int, int, int]: # Calculates the cost of using the network
    """Return (stored weights, MACs, paper-style weighted cost)."""
    dims = [784] + [nodes_per_layer] * num_layers + [10] # Creates the dimentions of the network as an array
    macs = sum(a * b for a, b in zip(dims, dims[1:])) # Used the dims array to calculate the neighboring layers
    weights = sum(a * b for a, b in zip(dims, dims[1:]))
    weight_unit_cost = 139
    multiplication_unit_cost = 1
    cost = weights * weight_unit_cost + macs * multiplication_unit_cost
    return weights, macs, cost


def train_and_evaluate(model, train_loader, test_loader, device, epochs, learning_rate): # Trains and then tests the trained model
    model.to(device) # Loads model
    loss_fn = nn.CrossEntropyLoss() # Creates the loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Creates the SGD optimizer which updates the weights of the NN so the prediction error becomes smaller
    for _ in range(epochs): # Number of training loops
        model.train() # Model in training mode
        for x, y in train_loader: # X is the images, Y is the correct answers/labels
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True) # Clears optimizer gradient
            loss_fn(model(x), y).backward() # model(x) preforms the forward pass over the data, then loss_fn(model(x), y) calculates how wrong the predictions are, then .backward() calculates the gradient of the loss
            optimizer.step() # Updates the weights based on the gradient of loss
    model.eval() # MOdel in evaluation mode
    correct = total = 0
    with torch.no_grad(): # Turns off gradient calculation
        for x, y in test_loader:
            logits = model(x.to(device)) # Sends the test images through the trained model
            correct += (logits.argmax(dim=1) == y.to(device)).sum().item() # The output scores are logits, we get their max which is the prediction then compare it to the correct answer, if its the same then we count it as a correct prediction
            total += y.numel()
    return 100.0 * correct / total # Calculates the accuracy 


def pareto_front(rows): # Finds the best trade offs between our parameters
    front = []
    for row in rows:
        dominated = any( # For model configuation it checks if there is an existing model better than it in all parameters, if true then the model is removed
            other["cost"] <= row["cost"]
            and other["accuracy_drop"] <= row["accuracy_drop"]
            and (other["cost"] < row["cost"] or other["accuracy_drop"] < row["accuracy_drop"])
            for other in rows if other is not row
        )
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda r: r["cost"])


def save_plot(rows, front, output: Path): # Creates a cost vs accuracy graph and saves it as an image
    plt.figure(figsize=(8, 5.5))
    plt.scatter([r["cost"] for r in rows], [r["accuracy_drop"] for r in rows],
                color="#8094a8", alpha=0.8, label="Explored configuration")
    plt.scatter([r["cost"] for r in front], [r["accuracy_drop"] for r in front],
                color="#d95f02", s=70, label="Pareto-optimal")
    if len(front) > 1:
        plt.plot([r["cost"] for r in front], [r["accuracy_drop"] for r in front],
                 color="#d95f02", linewidth=1.5)
    for r in rows:
        plt.annotate(f"({r['layers']},{r['width']})", (r["cost"], r["accuracy_drop"]),
                     xytext=(4, 3), textcoords="offset points", fontsize=7)
    plt.xlabel("Inference cost (normalized units)")
    plt.ylabel("Accuracy drop (%)")
    plt.title("MNIST MLP design-space exploration")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser() # Command line options
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=2) # How many times we go throught the entire dataset
    parser.add_argument("--batch-size", type=int, default=256) # Training batch size (how many images go through the training loop)
    parser.add_argument("--lr", type=float, default=0.01) # The learning rate decides how much we update the weights based on the gradient
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True) # Create output directory
    set_seed() # Sets initial seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Checks if cuda GPU exists, if not then it trains it on CPU
    print(f"device={device}")
    x_train, y_train, x_test, y_test = load_mnist(args.data_dir) # Load MNIST dataset
    train_loader = DataLoader( # Create the object that provides the training data in batches
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), # The arrays are converted into PyTorch tensors 
        batch_size=args.batch_size, shuffle=True, # Set batch size
        generator=torch.Generator().manual_seed(SEED), # Creates a PyTorch RNG with the seed
    )
    test_loader = DataLoader( # Create the test dataloader
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), # Convert test images and lables into PyTorch tensors and pair them
        batch_size=1024, shuffle=False, # Larger batch here since we dont optimize/edit weights
    )
    rows = [] # Create results list
    for layers in (1, 2, 3): # Starts the outer loop and tests networks with 1, 2 and 3 hidden laters
        for width in (10, 20, 40, 80, 160, 200): # Starts the inner loop and tests each network with 10, 20, 40, 80, 160, 200 width
            set_seed(SEED + layers * 1000 + width) # Sets seed for current configuration
            params, macs, cost = cost_model(layers, width) # Calculate cost
            start = time.time() # Records start time
            accuracy = train_and_evaluate( # Builds, trans and tests the model
                build_model(layers, width), train_loader, test_loader,
                device, args.epochs, args.lr,
            )
            row = {"layers": layers, "width": width, "weights": params, # Store the results of the model
                   "macs": macs, "cost": cost, "accuracy": accuracy,
                   "accuracy_drop": 100.0 - accuracy, "epochs": args.epochs,
                   "batch_size": args.batch_size, "learning_rate": args.lr,
                   "device": str(device), "seconds": time.time() - start}
            rows.append(row)
            print(f"n={layers} m={width}: accuracy={accuracy:.2f}% cost={cost} ({row['seconds']:.1f}s)")
    front = pareto_front(rows) # Removes all the bad/dominated model configuations
    for filename, selected in (("results.csv", rows), ("pareto_front.csv", front)): # Save into CSV file
        with (args.output_dir / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=selected[0].keys())
            writer.writeheader(); writer.writerows(selected)
    save_plot(rows, front, args.output_dir / "pareto_front.png") # Creates the plot and saves the image
    print("Pareto front:")
    for row in front:
        print(f"  (n={row['layers']}, m={row['width']}): cost={row['cost']}, drop={row['accuracy_drop']:.2f}%")


if __name__ == "__main__":
    main()
