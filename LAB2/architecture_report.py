"""Create copyable layer tables and receptive-field growth for the three models."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from models import build_model
from utils import count_trainable_parameters


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


def _layer_description(layer: nn.Module) -> tuple[str, str, str, str, str]:
    if isinstance(layer, nn.Conv2d):
        return (
            f"{layer.kernel_size[0]}x{layer.kernel_size[1]}",
            f"{layer.stride[0]}x{layer.stride[1]}",
            f"{layer.padding[0]}x{layer.padding[1]}",
            str(layer.in_channels),
            str(layer.out_channels),
        )
    if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
        kernel = _pair(layer.kernel_size)
        stride = _pair(layer.stride if layer.stride is not None else layer.kernel_size)
        padding = _pair(layer.padding)
        return (
            f"{kernel[0]}x{kernel[1]}",
            f"{stride[0]}x{stride[1]}",
            f"{padding[0]}x{padding[1]}",
            "spatial",
            "spatial",
        )
    if isinstance(layer, nn.Linear):
        return ("-", "-", "-", str(layer.in_features), str(layer.out_features))
    raise TypeError(f"Unsupported report layer: {type(layer).__name__}")


def architecture_rows(model: nn.Module) -> list[dict[str, object]]:
    """Run one dummy image through reportable layers and capture their dimensions."""
    rows: list[dict[str, object]] = []
    handles = []

    def hook(name: str, layer: nn.Module):
        def capture(_: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            kernel, stride, padding, in_channels, out_channels = _layer_description(layer)
            if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
                in_channels = str(inputs[0].shape[1])
                out_channels = str(output.shape[1])
            rows.append(
                {
                    "name": name,
                    "type": type(layer).__name__,
                    "input": tuple(inputs[0].shape),
                    "kernel": kernel,
                    "stride": stride,
                    "padding": padding,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "output": tuple(output.shape),
                    "parameters": sum(parameter.numel() for parameter in layer.parameters() if parameter.requires_grad),
                    "layer": layer,
                }
            )
        return capture

    report_types = (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.Linear)
    for name, layer in model.named_modules():
        if isinstance(layer, report_types):
            handles.append(layer.register_forward_hook(hook(name, layer)))
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 32, 32))
    for handle in handles:
        handle.remove()
    if tuple(output.shape) != (1, model.classifier[-1].out_features):
        raise RuntimeError(f"Unexpected model output shape: {tuple(output.shape)}")
    return rows


def receptive_field_rows(rows: list[dict[str, object]]) -> list[tuple[str, int, int]]:
    """Track receptive field and jump through convolution/pooling layers only."""
    receptive_field = 1
    jump = 1
    values = []
    for row in rows:
        layer = row["layer"]
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            kernel = _pair(layer.kernel_size)[0]
            stride = _pair(layer.stride if layer.stride is not None else layer.kernel_size)[0]
            receptive_field += (kernel - 1) * jump
            jump *= stride
            values.append((str(row["name"]), receptive_field, jump))
    return values


def format_report(model_name: str, num_classes: int) -> str:
    model = build_model(model_name, num_classes)
    rows = architecture_rows(model)
    learned_layers = sum(isinstance(row["layer"], (nn.Conv2d, nn.Linear)) for row in rows)
    lines = [
        f"{model_name.upper()} for {num_classes} classes",
        "Depth counts only learned CONV and FC layers; pooling and activations have no trainable weights.",
        f"Learned-layer depth: {learned_layers}",
        f"Trainable parameters: {count_trainable_parameters(model):,}",
        "",
        "Layer | Type | Input | Kernel | Stride | Padding | In channels/features | Out channels/features | Output | Parameters",
        "-" * 150,
    ]
    for row in rows:
        lines.append(
            " | ".join(
                str(row[field])
                for field in ("name", "type", "input", "kernel", "stride", "padding", "in_channels", "out_channels", "output", "parameters")
            )
        )
    lines.extend(["", "Receptive field growth (spatial CONV/pooling layers)", "Layer | Receptive field | Effective stride"])
    lines.extend(f"{name} | {field}x{field} | {jump}" for name, field, jump in receptive_field_rows(rows))
    return "\n".join(lines) + "\n"


def write_architecture_report(model_name: str, num_classes: int, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"architecture_summary_{model_name}_{num_classes}classes.txt"
    path.write_text(format_report(model_name, num_classes), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a Lab 2 layer-by-layer architecture summary.")
    parser.add_argument("--model", choices=["lenet", "alexnet", "vgg16"], required=True)
    parser.add_argument("--num-classes", type=int, choices=[10, 100], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    path = write_architecture_report(args.model, args.num_classes, args.output_dir)
    print(f"Saved architecture summary to {path}")


if __name__ == "__main__":
    main()
