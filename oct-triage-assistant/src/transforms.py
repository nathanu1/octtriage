"""
Image preprocessing and light augmentation.

Transfer learning tip: torchvision ResNet weights pretrained on ImageNet expect
RGB inputs scaled and normalized like ImageNet. Even when OCT is often grayscale,
we duplicate the channel to 3 planes so the backbone receives a tensor it was
designed for. Alternative: use single-channel models later for a research upgrade.
"""

from __future__ import annotations

from torchvision import transforms


def _imagenet_normalize():
    # Mean/std from ImageNet — paired with default torchvision ResNet weights.
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_train_transforms(img_size: int = 224):
    """
    Training transforms: resize + light randomness to reduce overfitting.

    RandomResizedCrop + horizontal flip are mild; for OCT you may later tune or
    remove flips if they are not justified anatomically.
    """
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            _imagenet_normalize(),
        ]
    )


def get_val_transforms(img_size: int = 224):
    """Validation / test: deterministic resize, no random augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            _imagenet_normalize(),
        ]
    )
