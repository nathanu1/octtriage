from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class OCTManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        split: str,
        transform,
        classes: list[str],
    ) -> None:
        frame = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        self.frame = frame[frame["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No samples found for split {split!r}")
        self.transform = transform
        self.classes = list(classes)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        unknown = set(self.frame["label"]) - set(self.classes)
        if unknown:
            raise ValueError(f"Manifest contains unknown labels: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        tensor = self.transform(image)
        return tensor, self.class_to_idx[row["label"]]
