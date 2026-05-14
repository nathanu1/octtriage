"""
Classifier model definition using transfer learning.

Idea: reuse a network (ResNet18) already trained on ImageNet as a **feature
extractor**, then train a small linear **head** on top for our four OCT
categories. Faster convergence than training a huge model from random init.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models


def build_model(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    ResNet18 with the final fully-connected layer replaced for `num_classes`.

    `pretrained=True` loads ImageNet weights (requires compatible torchvision).
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)

    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, num_classes)
    return net


def load_trained_weights(model: nn.Module, checkpoint_path: str | Path, map_location: str | torch.device) -> nn.Module:
    """Load `state_dict` from a training checkpoint onto an already constructed model."""
    ckpt = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model
