from dataclasses import dataclass
from typing import TypedDict, Union, Self
from engine.datasets_stats import DatasetStats
from engine.phoneme_indexes import PhonemeIndexes
from engine.normalization import zscore_normalize_with_uncond
from engine.config.train import TrainConfig
from engine.types.dataset import X, Y
import torch


class EncodedPhonemePart(TypedDict):
    id: str
    x: X
    y: Y

@dataclass
class GroupedMutePart:
    id: str
    duration_ms: float


@dataclass
class GroupedPhonemePart:
    id: str
    phonemes: list[str]         # Frame-level phonemes
    duration_ms: float
    pitches: list[int]          # Frame-level pitches
    mel_path: str
    mel_frames: int             # Number of mel frames

    # ----------------------------------------------------------
    def encode(
        self,
        dataset_stats: DatasetStats,
        phoneme_indexes: PhonemeIndexes,
        next_part: Self | GroupedMutePart | None,
    ) -> EncodedPhonemePart:
        """
        Frame-level encoding:
        - phonemes and pitches are already frame-aligned
        - No duration normalization needed since data is frame-level
        """
        pitch_stats = dataset_stats.get("pitches")
        mel_stats   = dataset_stats.get("mel")

        indexed_phonemes = [phoneme_indexes.index(p) for p in self.phonemes]

        next_mel_duration_ms = torch.tensor(next_part.duration_ms if isinstance(next_part, GroupedMutePart) else 0, dtype=torch.int32)

        x: X = {
            "phoneme_indexes": torch.tensor(indexed_phonemes, dtype=torch.int32),     # [T]
            "pitch": zscore_normalize_with_uncond(
                torch.tensor(self.pitches, dtype=torch.float32),                     # [T]
                pitch_stats["mean"], pitch_stats["std"], TrainConfig.get_pitch_uncond_token()
            ),
            "mel_frames": torch.tensor(self.mel_frames, dtype=torch.int32),              # mel_frames
            "next_mel_mute_duration_ms": next_mel_duration_ms,
        }

        # ---- mel -------------------------------------------------
        mel = torch.load(self.mel_path).squeeze(0)            # [80, T]
        mel = zscore_normalize_with_uncond(mel, mel_stats["mean"], mel_stats["std"], TrainConfig.get_mel_uncond_token())

        y: Y = {
            "mel":  mel,                                      # [80, T]  
            "mask": torch.ones_like(mel),                     # [80, T]  全て valid
        }

        return {"id": self.id, "x": x, "y": y}
    
    @staticmethod
    def get_uncond(pitch_shape: torch.Size) -> X:

        return {
            "phoneme_indexes": torch.full((1,), TrainConfig.phoneme_uncond_token,  dtype=torch.int32),
            "pitch":           torch.full(pitch_shape, TrainConfig.get_pitch_uncond_token(),  dtype=torch.float32),
            "mel_frames":          torch.tensor(1, dtype=torch.int32),
            "next_mel_mute_duration_ms": torch.tensor(0),
        }



@dataclass
class Metadata:
    current: Union[GroupedPhonemePart, GroupedMutePart, None]
    next: Union[GroupedPhonemePart, GroupedMutePart, None]
    prev: Union[GroupedPhonemePart, GroupedMutePart, None]