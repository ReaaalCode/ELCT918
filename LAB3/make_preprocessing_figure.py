"""Create a labeled preprocessing figure from a real handwritten digit photo."""
from __future__ import annotations

import argparse

import cv2
import matplotlib.pyplot as plt

from config import RESULTS_DIR
from preprocess import PreprocessResult, preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize all Lab 3 preprocessing stages from a real image.")
    parser.add_argument("--image", required=True, help="Path to a real handwritten-digit photo.")
    parser.add_argument("--dilate", action="store_true", help="Enable one small dilation step.")
    parser.add_argument("--min-contour-area", type=float, default=80.0)
    parser.add_argument("--output", default=str(RESULTS_DIR / "preprocessing_steps.png"))
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    result = preprocess(image, dilate=args.dilate, min_contour_area=args.min_contour_area, return_debug=True)
    if result is None:
        raise RuntimeError("No valid digit contour was found. Try a clearer image or lower --min-contour-area.")
    assert isinstance(result, PreprocessResult)
    figure, axes = plt.subplots(3, 3, figsize=(10, 10))
    for axis, (title, stage) in zip(axes.flat, result.stages.items()):
        if stage.ndim == 3:
            axis.imshow(cv2.cvtColor(stage, cv2.COLOR_BGR2RGB))
        else:
            axis.imshow(stage, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("MNIST-style preprocessing of a real handwritten digit", fontsize=14)
    figure.tight_layout()
    output = __import__("pathlib").Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"Saved preprocessing figure to {output}")


if __name__ == "__main__":
    main()
