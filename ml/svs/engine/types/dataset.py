from typing import TypedDict

import torch


class X(TypedDict):
    phoneme_indexes: torch.Tensor  # Frame-level phoneme indexes [T]
    pitch: torch.Tensor           # Frame-level pitch values [T]
    mel_frames: torch.Tensor          # Mel frames count (scalar)
    next_mel_mute_duration_ms: torch.Tensor


class Y(TypedDict):
    mel: torch.Tensor
    mask: torch.Tensor


class Item(TypedDict):
    x: X
    y: Y
