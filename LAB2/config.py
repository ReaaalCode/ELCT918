"""Shared defaults for the Lab 2 comparison."""
from pathlib import Path

SEED = 918
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
TRAIN_FRACTION = 0.9

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
PLOTS_DIR = PROJECT_DIR / "plots"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"

# CIFAR's official channel statistics. The same dataset-specific values are
# used for every architecture, so normalization cannot affect the comparison.
CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}

NUM_CLASSES = {"cifar10": 10, "cifar100": 100}
