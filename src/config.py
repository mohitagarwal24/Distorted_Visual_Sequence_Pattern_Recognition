"""Central configuration for the distorted-text recognition project.

Every tunable value lives here so experiments are reproducible and easy to
adjust without touching the rest of the code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

# Character set discovered during EDA (digits + uppercase letters, with the
# ambiguous 0/1/I/L/O removed). Index 0 is reserved for the CTC "blank" token,
# so real characters are mapped to indices 1..N.
CHARS = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"


@dataclass
class Config:
    # --- Paths ---
    data_root: Path = Path("data/cig_ps")
    train_dir: Path = field(init=False)
    test_dir: Path = field(init=False)
    labels_csv: Path = field(init=False)
    output_dir: Path = Path("outputs")

    # --- Image / preprocessing ---
    img_height: int = 32          # CRNN works on a fixed height; width carries the sequence
    img_width: int = 160
    chars: str = CHARS

    # --- Model ---
    rnn_hidden: int = 256         # hidden units per BiLSTM direction
    cnn_dropout: float = 0.1

    # --- Training ---
    batch_size: int = 64
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_fraction: float = 0.1
    num_workers: int = 0          # Windows + notebooks are safest with 0
    seed: int = 42
    max_train_samples: int | None = None  # set to an int for a quick smoke test

    # --- Augmentation toggles (help against noise / occlusion / blur) ---
    use_augmentation: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.train_dir = self.data_root / "train_images"
        self.test_dir = self.data_root / "test_images"
        self.labels_csv = self.data_root / "train-labels.csv"
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Number of model outputs = real characters + 1 blank slot.
    @property
    def num_classes(self) -> int:
        return len(self.chars) + 1
