"""CRNN: a CNN feature extractor followed by a BiLSTM sequence decoder.

The CNN turns the image into a horizontal sequence of feature vectors; the
BiLSTM reads that sequence left-to-right and right-to-left; a final linear
layer scores each character (plus a CTC blank) at every horizontal step.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes: int, img_height: int = 32, rnn_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        assert img_height == 32, "CNN stack is tuned for an input height of 32."

        def conv_block(in_c, out_c, bn=False):
            layers = [nn.Conv2d(in_c, out_c, 3, 1, 1), nn.ReLU(inplace=True)]
            if bn:
                layers.insert(1, nn.BatchNorm2d(out_c))
            return layers

        self.cnn = nn.Sequential(
            *conv_block(1, 64),
            nn.MaxPool2d(2, 2),                              # 32x160 -> 16x80
            *conv_block(64, 128),
            nn.MaxPool2d(2, 2),                              # -> 8x40
            *conv_block(128, 256, bn=True),
            *conv_block(256, 256),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),            # -> 4x41 (keep width)
            *conv_block(256, 512, bn=True),
            *conv_block(512, 512),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),            # -> 2x42
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(inplace=True),  # -> 1x41
            nn.Dropout(dropout),
        )

        self.rnn = nn.LSTM(512, rnn_hidden, num_layers=2, bidirectional=True, dropout=dropout, batch_first=False)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)                       # (B, C, 1, W)
        b, c, h, w = feat.size()
        assert h == 1, f"Expected feature height 1, got {h}."
        feat = feat.squeeze(2).permute(2, 0, 1)  # (W, B, C) — W is the time axis
        seq, _ = self.rnn(feat)
        logits = self.fc(seq)                    # (W, B, num_classes)
        return logits.log_softmax(2)             # CTCLoss expects log-probabilities
