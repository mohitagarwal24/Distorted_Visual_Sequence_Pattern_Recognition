"""Mapping between text labels and integer index sequences for CTC.

Index 0 is the CTC blank. Real characters occupy indices 1..N.
"""
from __future__ import annotations

from typing import List


class CharsetCodec:
    def __init__(self, chars: str):
        self.chars = chars
        self.blank = 0
        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(chars)}

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx[c] for c in text]

    def decode(self, indices: List[int]) -> str:
        """Collapse a raw CTC output: merge repeats, then drop blanks."""
        out, prev = [], None
        for idx in indices:
            if idx != prev and idx != self.blank:
                out.append(self.idx_to_char.get(idx, ""))
            prev = idx
        return "".join(out)
