# ELCT918 Lab 3 - Real-Time Handwritten Digit Recognition

This project trains the Lab 2 LeNet-5 structure from scratch on MNIST, then uses the saved weights in a webcam/video OCR pipeline. The camera image is converted into an MNIST-like digit before inference.

## Setup

Use Python 3.12 (the environment used here) and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The exact MNIST normalization is mean `0.1307`, standard deviation `0.3081`. MNIST is converted to a tensor, zero-padded from 28x28 to 32x32, then normalized. The webcam pipeline uses the same order and values.

## Train and evaluate

```bash
python train_mnist.py
python evaluate_mnist.py
```

Training saves the best-validation `state_dict` at `weights/lenet_mnist.pt`. The official MNIST test set is only evaluated after that checkpoint is reloaded.

## Preprocessing figure from a real photo

Take a photo of one handwritten digit on white paper, then run:

```bash
python make_preprocessing_figure.py --image path/to/my_digit_photo.jpg
```

This saves `results/preprocessing_steps.png`. For thin strokes, add `--dilate`.

## Real-time recognition

Laptop webcam (camera index 0):

```bash
python realtime_digit.py --source 0
```

Try another camera/phone-webcam index:

```bash
python realtime_digit.py --source 1
python realtime_digit.py --source 2
```

Video file instead of a camera:

```bash
python realtime_digit.py --source path/to/demo.mp4
```

The main window shows the centre ROI, predicted digit, confidence, and FPS. The second window shows the enlarged 28x28 MNIST-style image. Press `q` to quit cleanly.

Put a 30-60 second demo recording in a separate release/cloud link or link it here; temporary video files are ignored by Git.
