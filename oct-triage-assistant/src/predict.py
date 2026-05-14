"""
Single-image inference helper.

Given a saved checkpoint and one image path, run a forward pass and return
softmax probabilities. This is the same computation the Streamlit app uses,
but without the UI — useful for quick checks from the terminal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from src import config
from src.model import build_model, load_trained_weights
from src.transforms import get_val_transforms


def load_image_rgb(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


@torch.inference_mode()
def predict_image(
    model: torch.nn.Module,
    image_input: Union[Path, Image.Image],
    transform,
    device: torch.device,
    class_names: list[str],
) -> tuple[str, float, dict[str, float]]:
    if isinstance(image_input, Path):
        img = load_image_rgb(image_input)
    else:
        img = image_input.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    conf, idx = torch.max(probs, dim=0)
    label = class_names[int(idx)]
    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return label, float(conf), prob_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one OCT image (educational).")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file.")
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_PATH))
    parser.add_argument("--device", type=str, default=config.DEVICE)
    args = parser.parse_args()

    image_path = Path(args.image)
    ckpt_path = Path(args.checkpoint)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train with `python -m src.train` first."
        )

    device = torch.device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes: list[str] = list(ckpt.get("classes", list(config.CLASSES)))

    tfm = get_val_transforms(config.IMG_SIZE)
    model = build_model(num_classes=len(classes), pretrained=False)
    load_trained_weights(model, ckpt_path, map_location=device)
    model.to(device)

    label, conf, probs = predict_image(model, image_path, tfm, device, classes)

    print("\nEducational prediction (not a diagnosis):")
    print(f"  Predicted triage-style class: {label}")
    print(f"  Confidence (max softmax prob): {conf:.4f}")
    print("  Class probabilities:")
    for c, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"    {c}: {p:.4f}")


if __name__ == "__main__":
    main()
