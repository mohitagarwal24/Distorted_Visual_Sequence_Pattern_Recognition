"""Decoding, metrics, and reproducibility helpers."""
from __future__ import annotations

import random
from typing import List

import numpy as np
import torch

from .charset import CharsetCodec

try:
    import editdistance  # fast C implementation
    def _edit(a: str, b: str) -> int:
        return editdistance.eval(a, b)
except Exception:  # pragma: no cover - pure-python fallback
    def _edit(a: str, b: str) -> int:
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                cur = dp[j]
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
                prev = cur
        return dp[n]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def greedy_decode(log_probs: torch.Tensor, codec: CharsetCodec) -> List[str]:
    """Argmax at each time step, then collapse to text (CTC greedy decoding)."""
    best = log_probs.argmax(2).permute(1, 0)  # (B, W)
    return [codec.decode(seq.tolist()) for seq in best]


def char_error_rate(preds: List[str], targets: List[str]) -> float:
    """Mean Levenshtein distance normalised by target length (the PS metric)."""
    total_dist, total_len = 0, 0
    for p, t in zip(preds, targets):
        total_dist += _edit(p, t)
        total_len += max(len(t), 1)
    return total_dist / max(total_len, 1)
