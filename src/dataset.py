"""Dataset and image transforms for distorted grayscale text images."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .charset import CharsetCodec


def build_transforms(height: int, width: int, augment: bool) -> transforms.Compose:
    """Grayscale -> fixed size -> tensor -> normalised to roughly [-1, 1].

    Light augmentation (small rotations/translations) mimics the irregular
    alignment seen in the data and improves robustness.
    """
    ops: list = [transforms.Grayscale(num_output_channels=1)]
    if augment:
        ops.append(
            transforms.RandomAffine(degrees=4, translate=(0.03, 0.06), scale=(0.92, 1.0))
        )
    ops += [
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
    return transforms.Compose(ops)


class TextImageDataset(Dataset):
    """Returns (image_tensor, target_indices, target_length, filename)."""

    def __init__(
        self,
        image_dir: Path,
        codec: CharsetCodec,
        records: Optional[List[Tuple[str, str]]] = None,
        filenames: Optional[List[str]] = None,
        transform=None,
    ):
        self.image_dir = Path(image_dir)
        self.codec = codec
        self.transform = transform
        # Training mode supplies (filename, label) records; test mode supplies filenames only.
        if records is not None:
            self.items = records
            self.has_labels = True
        else:
            self.items = [(f, "") for f in (filenames or [])]
            self.has_labels = False

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        fname, label = self.items[i]
        img = Image.open(self.image_dir / fname)
        if self.transform:
            img = self.transform(img)
        target = torch.tensor(self.codec.encode(label), dtype=torch.long)
        return img, target, torch.tensor(len(target), dtype=torch.long), fname


def ctc_collate(batch):
    """Stack images and concatenate variable-length targets for CTCLoss."""
    imgs, targets, lengths, names = zip(*batch)
    imgs = torch.stack(imgs, 0)
    targets = torch.cat(targets) if targets[0].numel() or len(targets) else torch.tensor([], dtype=torch.long)
    lengths = torch.stack(lengths)
    return imgs, targets, lengths, list(names)


def load_clean_labels(labels_csv: Path) -> pd.DataFrame:
    """Read labels and drop the handful of Excel-corrupted rows (dates / sci-notation)."""
    df = pd.read_csv(labels_csv)
    df = df[["image", "text"]].copy()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.fullmatch(r"[23456789A-HJ-NP-Z]+")]
    return df.reset_index(drop=True)
