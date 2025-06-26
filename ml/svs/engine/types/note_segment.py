from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch

from engine.phoneme_indexes import PhonemeIndexes
from engine.config.train import TrainConfig


@dataclass
class PhonemeContext:
    prev2: str
    prev1: str
    current: str
    next1: str
    next2: str
    next3: str


    @staticmethod
    def get_uncond_value() -> torch.Tensor:
        return torch.tensor(
            [
                TrainConfig.phoneme_uncond_token,
                TrainConfig.phoneme_uncond_token,
                TrainConfig.phoneme_uncond_token,
                TrainConfig.phoneme_uncond_token,
                TrainConfig.phoneme_uncond_token,
                TrainConfig.phoneme_uncond_token
            ],
            dtype=torch.int64,
        )


    def to_tensor(self, phoneme_indexes: PhonemeIndexes) -> torch.Tensor:
        return torch.tensor(
            [
                phoneme_indexes.index(self.prev2),
                phoneme_indexes.index(self.prev1),
                phoneme_indexes.index(self.current),
                phoneme_indexes.index(self.next1),
                phoneme_indexes.index(self.next2),
                phoneme_indexes.index(self.next3),
            ],
            dtype=torch.int64,
        )


NULL_VAL = -1.0  # 無効値は明確な定数で（floatでも誤差なし）


@dataclass
class PitchContext:
    prev2: Optional[int]
    prev1: Optional[int]
    current: Optional[int]
    next1: Optional[int]
    next2: Optional[int]
    next3: Optional[int]


    @staticmethod
    def get_uncond_value() -> torch.Tensor:
        pitch_uncond = TrainConfig.get_pitch_uncond_token()
        return torch.tensor(
            [
                pitch_uncond,
                pitch_uncond,
                pitch_uncond,
                pitch_uncond,
                pitch_uncond,
                pitch_uncond
            ],
            dtype=torch.float32
        )

    def to_tensor(
        self,
        *,
        null_value: float = NULL_VAL,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = torch.tensor(
            [
                v if v is not None else null_value
                for v in (
                    self.prev2,
                    self.prev1,
                    self.current,
                    self.next1,
                    self.next2,
                    self.next3,
                )
            ],
            dtype=torch.float32,
        )
        mask = (raw != null_value).float()
        return raw, mask

    def normalize(
        self,
        mean: int | float | torch.Tensor,
        std: int | float | torch.Tensor,
        *,
        null_value: float = NULL_VAL,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw, mask = self.to_tensor(null_value=null_value)

        mean_t = torch.as_tensor(mean, dtype=raw.dtype, device=raw.device)
        std_t = torch.as_tensor(std, dtype=raw.dtype, device=raw.device)

        std_safe = torch.where(std_t.abs() < 1e-8, torch.ones_like(std_t), std_t)

        z = (raw - mean_t) / std_safe
        z[mask == 0] = null_value

        return z, mask


@dataclass
class MuteSegment:
    type: Literal["MUTE"]
    id: str
    file: str
    start_time: int
    end_time: int


@dataclass
class PhonemeSegment:
    type: Literal["PHONEME"]
    id: str
    file: str
    phoneme_context: PhonemeContext
    pitch_context: PitchContext
    pitch_name: str
    notenum: int
    start_time: int
    end_time: int


NoteSegment = Union[MuteSegment, PhonemeSegment]
