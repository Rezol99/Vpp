import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import torch.nn.functional as F

from engine.config.train import TrainConfig
from engine.datasets_stats import DatasetStats
from engine.phoneme_indexes import PhonemeIndexes
from engine.types.grouped_part import GroupedPhonemePart, GroupedMutePart
from engine.types.dataset import X, Y
from engine.grouped_parts import GroupedParts
import logging
from typing import Tuple
import os

class VocalDataset(Dataset):
    @staticmethod
    def _resolve_cache_path(small: bool, prediction: bool) -> str:
        if small:
            return "./_cache/small_dataset_cache.pt"
        elif prediction:
            return "./_cache/prediction_dataset_cache.pt"
        return "./_cache/dataset_cache.pt"
    
    def __init__(
        self,
        grouped_parts: GroupedParts,
        dataset_stats: DatasetStats,
        phoneme_indexes: PhonemeIndexes,
        small: bool = False,
        prediction: bool = False,
    ):
        cache_path = VocalDataset._resolve_cache_path(small=small, prediction=prediction)

        if os.path.exists(cache_path):
            logging.info("Loading cached dataset...")
            cache = torch.load(cache_path, weights_only=True)
            self.x: list = cache["x"]
            self.y: list = cache["y"]
        else:
            mode = "small " if small else "prediction " if prediction else ""
            logging.info(f"Building {mode}dataset from scratch...")

            all_metadata = grouped_parts.get_all_metadata()
            all_metadata = all_metadata[:100] if small else all_metadata

            encoded_x: list[X] = []
            encoded_y: list[Y] = []
            skipped = 0

            for metadata in tqdm(all_metadata, desc="Initializing dataset"):
                current_data = metadata.current
                next_data = metadata.next

                if not isinstance(current_data, GroupedPhonemePart):
                    continue

                try:
                    x_item, y_item = VocalDataset.encode(
                        current_data,
                        next_data,
                        dataset_stats,
                        phoneme_indexes,
                    )
                except Exception as e:
                    logging.warning(f"Skipping part {current_data.id}: {e}")
                    skipped += 1
                    continue

                encoded_x.append(x_item)
                encoded_y.append(y_item)

            logging.info(
                f"Skipped {skipped} samples. Kept {len(encoded_x)} samples."
            )

            self.x = encoded_x
            self.y = encoded_y

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({"x": self.x, "y": self.y}, cache_path)
            logging.info(f"Saved dataset cache at {cache_path}")

    @staticmethod
    def encode(
        part: GroupedPhonemePart,
        next_part: GroupedPhonemePart | GroupedMutePart | None,
        dataset_stats: DatasetStats,
        phoneme_indexes: PhonemeIndexes,
    ) -> Tuple[X, Y]:
        # GroupedPart から事前エンコード済みテンソルと mel_path を取得
        enc = part.encode(dataset_stats, phoneme_indexes, next_part)
        phoneme = enc["x"]["phoneme_indexes"]
        pitch = enc["x"]["pitch"]
        mel_frames = enc["x"]["mel_frames"]
        next_mel_mute_duration_ms = enc["x"]["next_mel_mute_duration_ms"]
        mel = enc["y"]["mel"]
        mask = enc["y"]["mask"]

        # Get current sequence length
        seq_len = phoneme.shape[0]
        max_length = TrainConfig.max_sequence_length
        
        # Skip samples that are too long (likely corrupted data)
        if seq_len > max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max_length {max_length}")
        
        # Pad sequences to max_length
        if seq_len < max_length:
            pad_len = max_length - seq_len
            
            # Pad phoneme_indexes with padding token
            phoneme = F.pad(phoneme, (0, pad_len), value=phoneme_indexes.pad_index)  # Use -1 as padding token
            
            # Pad pitch with padding value
            pitch = F.pad(pitch, (0, pad_len), value=float(TrainConfig.pad_value))
            
            # Pad mel spectrogram [80, T] -> [80, max_length] with silence value
            mel = F.pad(mel, (0, pad_len), value=TrainConfig.get_mel_uncond_token())
            
            # Create proper mask: 1 for valid frames, 0 for padded frames
            valid_mask = torch.ones(80, seq_len)
            pad_mask = torch.zeros(80, pad_len)
            mask = torch.cat([valid_mask, pad_mask], dim=1)

        x_item: X = {
            "phoneme_indexes": phoneme,
            "pitch": pitch,
            "mel_frames": torch.tensor(mel_frames, dtype=torch.int32),
            "next_mel_mute_duration_ms": next_mel_mute_duration_ms
        }

        y_item: Y = {
            "mel": mel,
            "mask": mask
        }

        return x_item, y_item

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching padded sequences."""
        x_batch, y_batch = zip(*batch)
        
        # Stack all tensors since they are now all the same size due to padding
        batched_x = {
            "phoneme_indexes": torch.stack([x["phoneme_indexes"] for x in x_batch]),
            "pitch": torch.stack([x["pitch"] for x in x_batch]),
            "mel_frames": torch.stack([x["mel_frames"] for x in x_batch]),
            "next_mel_mute_duration_ms": torch.stack([x["next_mel_mute_duration_ms"] for x in x_batch])
        }
        
        batched_y = {
            "mel": torch.stack([y["mel"] for y in y_batch]),
            "mask": torch.stack([y["mask"] for y in y_batch]),
        }
        
        return batched_x, batched_y
