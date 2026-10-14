"""Real-time Lab 3 digit recognition from a camera index or video file."""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from config import WEIGHTS_PATH
from models import LeNet5MNIST
from preprocess import PreprocessResult, preprocess


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def centered_roi(frame: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    actual_size = min(size, height, width)
    x = (width - actual_size) // 2
    y = (height - actual_size) // 2
    return frame[y:y + actual_size, x:x + actual_size], (x, y, actual_size, actual_size)


def load_model(device: torch.device) -> LeNet5MNIST:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}. Run train_mnist.py first.")
    model = LeNet5MNIST().to(device)
    try:
        state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lab 3 digit recognition on a webcam or video file.")
    parser.add_argument("--source", default="0", help="Camera index (0, 1, 2) or video file path.")
    parser.add_argument("--roi-size", type=int, default=320)
    parser.add_argument("--dilate", action="store_true", help="Use one small dilation step for thin strokes.")
    parser.add_argument("--min-contour-area", type=float, default=80.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    source = parse_source(args.source)
    # On this Linux laptop, explicitly selecting V4L2 avoids OpenCV choosing an
    # incompatible default backend for a numeric webcam index. File paths keep
    # OpenCV's normal video-file backend.
    capture = cv2.VideoCapture(source, cv2.CAP_V4L2) if isinstance(source, int) else cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source {args.source}. Try camera indices 0, 1, or 2, or a valid video path.")
    fps_times: deque[float] = deque(maxlen=20)
    last_time = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("No more frames or frame read failed.")
                break
            now = time.perf_counter()
            fps_times.append(now - last_time)
            last_time = now
            fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0
            roi, (x, y, width, height) = centered_roi(frame, args.roi_size)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            result = preprocess(roi, dilate=args.dilate, min_contour_area=args.min_contour_area, return_debug=True)
            text = "No digit detected"
            model_view = np.zeros((28, 28), dtype=np.uint8)
            if result is not None:
                assert isinstance(result, PreprocessResult)
                with torch.no_grad():
                    probabilities = torch.softmax(model(result.tensor.to(device)), dim=1)[0]
                confidence, digit = probabilities.max(dim=0)
                text = f"Digit: {digit.item()}  Confidence: {confidence.item() * 100:.1f}%"
                model_view = result.centered_28
            cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Lab 3 - Real-time digit recognition", frame)
            cv2.imshow("Model view - centered 28x28", cv2.resize(model_view, (280, 280), interpolation=cv2.INTER_NEAREST))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
