"""Generate the submission CSV from a trained checkpoint."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .charset import CharsetCodec
from .config import Config
from .dataset import TextImageDataset, build_transforms, ctc_collate
from .model import CRNN
from .utils import greedy_decode


def _natural_key(name: str):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else name


def load_model(checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device)
    codec = CharsetCodec(ckpt["cfg_chars"])
    model = CRNN(len(ckpt["cfg_chars"]) + 1, ckpt["img_height"], ckpt["rnn_hidden"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, codec, ckpt["img_height"], ckpt["img_width"]


@torch.no_grad()
def predict(cfg: Config, checkpoint: Path | None = None, out_name: str = "submission.csv") -> Path:
    device = torch.device(cfg.device)
    checkpoint = Path(checkpoint or cfg.output_dir / "crnn_best.pt")
    model, codec, h, w = load_model(checkpoint, device)

    filenames = sorted([p.name for p in cfg.test_dir.glob("*.png")], key=_natural_key)
    ds = TextImageDataset(cfg.test_dir, codec, filenames=filenames,
                          transform=build_transforms(h, w, augment=False))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=ctc_collate)

    rows = []
    for imgs, _, _, names in tqdm(loader, desc="Predicting"):
        preds = greedy_decode(model(imgs.to(device)).cpu(), codec)
        rows.extend(zip(names, preds))

    out_path = cfg.output_dir / out_name
    pd.DataFrame(rows, columns=["image", "prediction"]).to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    print(predict(Config()))
