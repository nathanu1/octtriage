from __future__ import annotations

from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import functional as functional


class ResizeWithPad:
    def __init__(self, size: int, fill: int = 0) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.size / max(width, height)
        resized = functional.resize(
            image,
            [max(1, round(height * scale)), max(1, round(width * scale))],
            antialias=True,
        )
        pad_w = self.size - resized.width
        pad_h = self.size - resized.height
        padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
        return functional.pad(resized, padding, fill=self.fill)


class MedianSpeckleReduction:
    def __call__(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.MedianFilter(size=3))


def _normalization() -> transforms.Normalize:
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def training_transform(image_size: int = 224):
    return transforms.Compose(
        [
            MedianSpeckleReduction(),
            ResizeWithPad(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.ToTensor(),
            _normalization(),
        ]
    )


def inference_transform(image_size: int = 224):
    return transforms.Compose(
        [
            MedianSpeckleReduction(),
            ResizeWithPad(image_size),
            transforms.ToTensor(),
            _normalization(),
        ]
    )
