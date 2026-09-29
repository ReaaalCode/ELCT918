# ELCT918 Lab 2

This lab compares custom PyTorch implementations of LeNet-5, AlexNet, and VGG16 on CIFAR-10 and CIFAR-100. All networks are trained from scratch. No pretrained weights or torchvision model implementations are used.

The same training settings and preprocessing are used for the three architectures within each dataset so the comparison is fair.

## Setup

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The code automatically uses CUDA when PyTorch can see a CUDA GPU. It prints the selected GPU at startup. `requirements.txt` pins the verified Linux CUDA 13.0 PyTorch and torchvision pair for this RTX 3060 Laptop GPU, so use the command above instead of installing CPU-only packages separately.

## Files and folders

```text
models.py                 Custom LeNet-5, AlexNet, and VGG16
train.py                  One reusable training command
evaluate.py               Training and evaluation loops
utils.py                  Data split, reproducibility, saving, and plots
config.py                 Common defaults
architecture_report.py    Layer table and receptive-field report
results/                  Per-epoch metrics, final metrics, and summary.csv
plots/                    Accuracy and loss curves
checkpoints/              Best validation checkpoint for each run
data/                     Downloaded CIFAR data, created when training starts
```

## Training settings

Default settings for every run are 20 epochs, batch size 128, SGD, learning rate 0.01, momentum 0.9, and weight decay `5e-4`. MultiStepLR reduces the learning rate by 10 times after epochs 10 and 15. The seed is 918.

The official training set is split with that seed into 45,000 training images and 5,000 validation images. The official test set is only used after training. Training uses random crop with padding 4, random horizontal flip, tensor conversion, and CIFAR normalization. Validation and test use tensor conversion and the same normalization.

## Run the six experiments

Run these commands from this folder:

```bash
python train.py --model lenet --dataset cifar10
python train.py --model alexnet --dataset cifar10
python train.py --model vgg16 --dataset cifar10

python train.py --model lenet --dataset cifar100
python train.py --model alexnet --dataset cifar100
python train.py --model vgg16 --dataset cifar100
```

Useful optional arguments are `--epochs`, `--batch-size`, `--lr`, and `--seed`.

For a one-epoch check before a full run:

```bash
python train.py --model lenet --dataset cifar10 --smoke-test
```

## CIFAR adaptations

- LeNet-5 changes the input from one channel to RGB and changes the final classifier to 10 or 100 classes. Its original 5x5 valid convolutions and average pooling still fit a 32x32 input. C5 is implemented as a full 16-to-120 convolution; C3 also uses full connectivity instead of LeNet's original partial connection pattern.
- AlexNet changes only its first convolution from 11x11, stride 4 to 5x5, stride 1 because the original setting is too aggressive for 32x32 images. It keeps the original 96/256/384/384/256 channel widths, grouped Conv2/Conv4/Conv5, LRN, dropout, 4096-wide hidden FC layers, and overlapping 3x3 stride-2 max pooling. CIFAR reaches a 256x3x3 map before FC1.
- VGG16 keeps all 13 convolution layers, 3x3 filters, five max-pooling layers, and three FC layers. A 32x32 input reaches a 1x1 map after the five pools, so only the first FC input changes from 25088 to 512; its two hidden classifier layers remain 4096-wide. No BatchNorm or residual connections are added.

VGG16 uses Kaiming initialization for convolutional ReLU layers and small-normal initialization for fully-connected layers to enable stable from-scratch training of the deep non-BatchNorm network.

The architecture report counts depth as learned CONV and FC layers only: LeNet-5 has 5, AlexNet has 8, and VGG16 has 16. It also lists receptive-field growth. Larger early filters in LeNet/AlexNet grow it quickly, while VGG16 grows it through several stacked 3x3 convolutions.

## Results

Each run saves per-epoch CSV and JSON metrics in `results/`, a final JSON file, a best-validation checkpoint, and these plots:

```text
plots/lenet_cifar10_accuracy.png
plots/lenet_cifar10_loss.png
```

The same naming is used for every architecture/dataset pair. `results/summary.csv` contains the final table for all six runs.

Top-1 accuracy means the class with the highest output score is correct. CIFAR-100 also records top-5 accuracy, where the correct class only needs to be among the five highest scores. Accuracy drop is `100 - test top-1 accuracy`.
