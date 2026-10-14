"""Non-camera sanity checks for the Lab 3 preprocessing module."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from config import MNIST_MEAN, MNIST_STD
from preprocess import PreprocessResult, preprocess


def synthetic_roi() -> np.ndarray:
    """Unit-test input only; it is not a substitute for a real report photograph."""
    image = np.full((180, 180, 3), 255, dtype=np.uint8)
    cv2.putText(image, "7", (48, 145), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 12, cv2.LINE_AA)
    return image


def main() -> None:
    result = preprocess(synthetic_roi(), return_debug=True)
    assert isinstance(result, PreprocessResult)
    assert tuple(result.tensor.shape) == (1, 1, 32, 32)
    assert result.centered_28.shape == (28, 28)
    assert result.padded_32.shape == (32, 32)
    assert torch.isfinite(result.tensor).all()
    assert max(result.resized_shape) == 20, result.resized_shape
    original_width, original_height = result.bounding_box[2], result.bounding_box[3]
    resized_height, resized_width = result.resized_shape
    assert abs((original_width / original_height) - (resized_width / resized_height)) < 0.12
    # The 0-valued padding must be normalized using the exact training values.
    assert torch.isclose(result.tensor[0, 0, 0, 0], torch.tensor(-MNIST_MEAN / MNIST_STD), atol=1e-6)
    blank = np.full((160, 160, 3), 255, dtype=np.uint8)
    assert preprocess(blank) is None
    dilated = preprocess(synthetic_roi(), dilate=True, return_debug=True)
    assert isinstance(dilated, PreprocessResult)
    print("PASS: shape, finite tensor, aspect ratio, centering stages, normalization, and no-contour handling.")


if __name__ == "__main__":
    main()
