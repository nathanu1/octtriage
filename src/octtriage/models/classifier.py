from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    classes: list[str]
    image_size: int
    backbone: str
    model_version: str
    checkpoint_path: Path


def build_classifier(
    num_classes: int,
    *,
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
) -> nn.Module:
    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported backbone: {backbone}")


def gradcam_target_layer(model: nn.Module, backbone: str) -> nn.Module:
    if backbone == "efficientnet_b0":
        return model.features[-1]  # type: ignore[attr-defined]
    if backbone == "resnet18":
        return model.layer4[-1]  # type: ignore[attr-defined]
    raise ValueError(f"Unsupported backbone: {backbone}")


def load_bundle(checkpoint_path: str | Path, device: torch.device) -> ModelBundle:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"model_state_dict", "classes", "image_size", "model_version", "backbone"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing metadata fields: {sorted(missing)}")
    model = build_classifier(
        len(checkpoint["classes"]),
        backbone=checkpoint["backbone"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return ModelBundle(
        model=model,
        classes=list(checkpoint["classes"]),
        image_size=int(checkpoint["image_size"]),
        backbone=str(checkpoint["backbone"]),
        model_version=str(checkpoint["model_version"]),
        checkpoint_path=path,
    )
