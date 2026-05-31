"""Training loop for the CRNN + CTC model."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .charset import CharsetCodec
from .config import Config
from .dataset import (
    TextImageDataset,
    build_transforms,
    ctc_collate,
    load_clean_labels,
)
from .model import CRNN
from .utils import char_error_rate, greedy_decode, set_seed


def build_dataloaders(cfg: Config, codec: CharsetCodec) -> Tuple[DataLoader, DataLoader]:
    df = load_clean_labels(cfg.labels_csv)
    if cfg.max_train_samples:
        df = df.sample(n=min(cfg.max_train_samples, len(df)), random_state=cfg.seed).reset_index(drop=True)

    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    n_val = int(len(df) * cfg.val_fraction)
    val_df, train_df = df.iloc[:n_val], df.iloc[n_val:]

    train_ds = TextImageDataset(
        cfg.train_dir, codec,
        records=list(zip(train_df["image"], train_df["text"])),
        transform=build_transforms(cfg.img_height, cfg.img_width, cfg.use_augmentation),
    )
    val_ds = TextImageDataset(
        cfg.train_dir, codec,
        records=list(zip(val_df["image"], val_df["text"])),
        transform=build_transforms(cfg.img_height, cfg.img_width, augment=False),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=ctc_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=ctc_collate)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, codec, device) -> Tuple[float, List[str], List[str]]:
    model.eval()
    preds, gts = [], []
    for imgs, targets, lengths, _ in loader:
        log_probs = model(imgs.to(device))
        preds.extend(greedy_decode(log_probs.cpu(), codec))
        offset = 0
        for n in lengths.tolist():
            chunk = targets[offset:offset + n].tolist()
            gts.append("".join(codec.idx_to_char[i] for i in chunk))
            offset += n
    return char_error_rate(preds, gts), preds, gts


def train(cfg: Config) -> Dict:
    set_seed(cfg.seed)
    codec = CharsetCodec(cfg.chars)
    device = torch.device(cfg.device)

    train_loader, val_loader = build_dataloaders(cfg, codec)
    model = CRNN(cfg.num_classes, cfg.img_height, cfg.rnn_hidden, cfg.cnn_dropout).to(device)

    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "val_cer": []}
    best_cer = float("inf")
    ckpt_path = cfg.output_dir / "crnn_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for imgs, targets, target_lengths, _ in pbar:
            imgs, targets, target_lengths = imgs.to(device), targets.to(device), target_lengths.to(device)
            log_probs = model(imgs)
            T, B, _ = log_probs.size()
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item() * B
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = running / len(train_loader.dataset)
        val_cer, _, _ = evaluate(model, val_loader, codec, device)
        scheduler.step(val_cer)
        history["train_loss"].append(train_loss)
        history["val_cer"].append(val_cer)
        print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | val_CER {val_cer:.4f}")

        if val_cer < best_cer:
            best_cer = val_cer
            torch.save({"model": model.state_dict(), "cfg_chars": cfg.chars,
                        "img_height": cfg.img_height, "img_width": cfg.img_width,
                        "rnn_hidden": cfg.rnn_hidden}, ckpt_path)

    history["best_cer"] = best_cer
    history["checkpoint"] = str(ckpt_path)
    return history


if __name__ == "__main__":
    print(train(Config()))
