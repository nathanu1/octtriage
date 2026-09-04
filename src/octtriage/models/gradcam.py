from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image


class GradCAM:
    """Generate a coarse class-activation map; this is not a causal explanation."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._capture_activations)
        self._backward_handle = target_layer.register_full_backward_hook(self._capture_gradients)

    def _capture_activations(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _capture_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(tensor)
            logits[:, class_index].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture model tensors")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        activation = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        activation = torch.nn.functional.interpolate(
            activation,
            size=tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        activation -= activation.min()
        maximum = activation.max()
        if maximum > 0:
            activation /= maximum
        return activation.detach().cpu().numpy()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    base = image.convert("RGB")
    map_image = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(base.size)
    values = np.asarray(map_image, dtype=np.float32) / 255.0
    colored = np.zeros((base.height, base.width, 3), dtype=np.uint8)
    colored[..., 0] = np.uint8(values * 255)
    colored[..., 1] = np.uint8(np.clip(1.0 - np.abs(values - 0.5) * 2.0, 0, 1) * 160)
    overlay = Image.fromarray(colored)
    return Image.blend(base, overlay, alpha=alpha)


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
