"""Shared camera-to-MNIST preprocessing for Lab 3."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from config import MNIST_MEAN, MNIST_STD


@dataclass
class PreprocessResult:
    """A valid model tensor plus images used to inspect every preprocessing step."""

    tensor: torch.Tensor
    centered_28: np.ndarray
    padded_32: np.ndarray
    stages: dict[str, np.ndarray]
    bounding_box: tuple[int, int, int, int]
    resized_shape: tuple[int, int]


def _find_contours(binary: np.ndarray) -> list[np.ndarray]:
    found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return found[0] if len(found) == 2 else found[1]


def _box_gap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    """Shortest distance between two axis-aligned bounding boxes."""
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    gap_x = max(ax - (bx + bw), bx - (ax + aw), 0)
    gap_y = max(ay - (by + bh), by - (ay + ah), 0)
    return float(np.hypot(gap_x, gap_y))


def _union_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    return x0, y0, x1 - x0, y1 - y0


def preprocess(
    roi: np.ndarray,
    *,
    dilate: bool = False,
    min_contour_area: float = 80.0,
    border_margin: int = 5,
    max_aspect_ratio: float = 6.0,
    component_gap_ratio: float = 0.20,
    return_debug: bool = False,
) -> torch.Tensor | PreprocessResult | None:
    """Convert a BGR camera ROI to normalized [1, 1, 32, 32] MNIST input.

    Returns None if no sufficiently large foreground contour is visible. Set
    return_debug=True to receive all reportable intermediate images.
    """
    if roi is None or roi.size == 0 or roi.ndim != 3 or roi.shape[2] != 3:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    processed = thresholded.copy()
    if dilate:
        processed = cv2.dilate(processed, np.ones((2, 2), dtype=np.uint8), iterations=1)
    contours = _find_contours(processed)
    image_height, image_width = processed.shape
    valid: list[np.ndarray] = []
    for candidate in contours:
        x, y, width, height = cv2.boundingRect(candidate)
        if cv2.contourArea(candidate) < min_contour_area:
            continue
        # Hands, page edges, and camera shadows commonly connect to an ROI edge.
        # A digit held inside the displayed centre box should not touch that edge.
        touched_edges = sum((
            x <= border_margin,
            y <= border_margin,
            x + width >= image_width - border_margin,
            y + height >= image_height - border_margin,
        ))
        # Reject page/hands that span a corner or multiple frame edges, but keep
        # a genuine digit that happens to touch one edge of a still photograph.
        if touched_edges >= 2:
            continue
        # Remove long ruled-paper lines while still allowing narrow digits such as 1.
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > max_aspect_ratio:
            continue
        valid.append(candidate)
    if not valid:
        return None
    # Start with the largest valid component, as specified. Attach only nearby
    # valid components so a handwritten digit with disconnected strokes (for
    # example, the loop and stem of a 9) remains one digit.
    boxes = [cv2.boundingRect(candidate) for candidate in valid]
    largest_index = max(range(len(valid)), key=lambda index: cv2.contourArea(valid[index]))
    selected = [boxes[largest_index]]
    changed = True
    while changed:
        changed = False
        combined = _union_box(selected)
        allowed_gap = max(10.0, component_gap_ratio * max(combined[2], combined[3]))
        for box in boxes:
            if box in selected or _box_gap(box, combined) > allowed_gap:
                continue
            selected.append(box)
            changed = True
    x, y, width, height = _union_box(selected)
    cropped = processed[y:y + height, x:x + width]
    if cropped.size == 0 or width <= 0 or height <= 0:
        return None

    scale = 20.0 / max(width, height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(cropped, (resized_width, resized_height), interpolation=interpolation)

    # Initial placement prevents clipping before shifting its centre of mass to (13.5, 13.5).
    canvas = np.zeros((28, 28), dtype=np.uint8)
    start_x = (28 - resized_width) // 2
    start_y = (28 - resized_height) // 2
    canvas[start_y:start_y + resized_height, start_x:start_x + resized_width] = resized
    moments = cv2.moments(canvas)
    if moments["m00"] != 0:
        centre_x = moments["m10"] / moments["m00"]
        centre_y = moments["m01"] / moments["m00"]
        shift = np.float32([[1, 0, 13.5 - centre_x], [0, 1, 13.5 - centre_y]])
        centered = cv2.warpAffine(canvas, shift, (28, 28), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        centered = canvas
    padded = cv2.copyMakeBorder(centered, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

    # Same numerical convention as training: raw 0..255 -> 0..1 -> normalize.
    tensor = torch.from_numpy(padded.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - MNIST_MEAN) / MNIST_STD
    if not torch.isfinite(tensor).all():
        raise RuntimeError("Preprocessing produced a non-finite tensor.")
    if not return_debug:
        return tensor
    stages = {
        "Original ROI": roi,
        "Grayscale": gray,
        "Gaussian blurred": blurred,
        "Thresholded + inverted": thresholded,
        "Dilated" if dilate else "Dilation disabled": processed,
        "Cropped digit": cropped,
        "Resized digit": resized,
        "Centered 28x28": centered,
        "Padded 32x32": padded,
    }
    return PreprocessResult(tensor, centered, padded, stages, (x, y, width, height), (resized_height, resized_width))
