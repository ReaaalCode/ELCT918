"""Shared reproducible settings for Lab 3."""
from pathlib import Path

SEED = 918
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
SCHEDULER_MILESTONES = [6, 8]
SCHEDULER_GAMMA = 0.1
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
WEIGHTS_PATH = PROJECT_DIR / "weights" / "lenet_mnist.pt"
RESULTS_DIR = PROJECT_DIR / "results"
