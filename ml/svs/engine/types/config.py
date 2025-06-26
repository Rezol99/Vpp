from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DistConfig:
    dist_backend: str
    dist_url: str
    world_size: int


@dataclass
class UpsampleConfig:
    rates: List[int]
    kernel_sizes: List[int]
    initial_channel: int


@dataclass
class ResblockConfig:
    kernel_sizes: List[int]
    dilation_sizes: List[List[int]]


@dataclass
class AudioConfig:
    segment_size: int
    num_mels: int
    num_freq: int
    n_fft: int
    hop_size: int
    win_size: int
    sampling_rate: int
    fmin: int
    fmax: int
    fmax_for_loss: Optional[float]


@dataclass
class Config:
    resblock: str
    num_gpus: int
    batch_size: int
    learning_rate: float
    adam_b1: float
    adam_b2: float
    lr_decay: float
    seed: int
    upsample: UpsampleConfig
    resblock_config: ResblockConfig
    audio: AudioConfig
    num_workers: int
    dist_config: DistConfig
