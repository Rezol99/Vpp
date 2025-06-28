import os
import shutil
import logging
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from engine.config.train import TrainConfig
from engine.datasets_stats import DatasetStats
from engine.phoneme_indexes import PhonemeIndexes
from engine.grouped_parts import GroupedParts
from engine.datasets.vocal import VocalDataset
from engine.models.svs import SVSModel, Inputs
from engine.diffuser import Diffuser
from engine.normalization import inverse_zscore_normalize_with_uncond
from engine.utils.audio import (
    save_audio,
    create_silent_mel_spectrogram,
    mel_spectrogram_to_audio,
)
from engine.utils.time import get_time_str
from engine.types.dataset import X

PREDICTION_BATCH_SIZE = 4
CFG_SCALE = 3

# TODO:　prevも考慮しないと初めに無音があったときの対応ができない
class Predictor:
    def __init__(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        dataset_stats = DatasetStats()
        phoneme_indexes = PhonemeIndexes()
        grouped_parts = GroupedParts(prediction=True)

        dataset = VocalDataset(
            grouped_parts=grouped_parts,
            dataset_stats=dataset_stats,
            phoneme_indexes=phoneme_indexes,
            prediction=True,
        )

        mel_stats = dataset_stats.get("mel")
        model = SVSModel(TrainConfig.phoneme_emb_dim, prediction=True).to(device)
        model.eval()
        checkpoint = model.load(checkpoint="./checkpoints/product/checkpoint.pth", map_location=device)

        diffuser = Diffuser(device=device, debug=True, fast_inference=True)

        loader = DataLoader(
            dataset,
            batch_size=PREDICTION_BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        self.model = model
        self.loader = loader
        self.diffuser = diffuser
        self.mel_stats = mel_stats
        self.checkpoint = checkpoint
        self.device = device

    def sample(self):
        mel_chunks: list[torch.Tensor] = []
        silence_chunks: list[int] = []
        silence_positions: list[int] = []
        
        with torch.no_grad():
            for batch in tqdm(self.loader, desc="Sampling"):
                x_batch, y_batch = batch
                x: X = {k: v.to(self.device) for k, v in x_batch.items()}  # type: ignore[arg-type]
                B, C, T_pad = y_batch["mel"].shape

                inputs: Inputs = {"x": x, "noisy_x": None, "t": None}  # type: ignore
                mel_shape = (B, C, T_pad)
                pitch_shape = x["pitch"].size()

                pred_batch: torch.Tensor = self.diffuser.sample(
                    self.model, inputs, mel_shape, pitch_shape, cfg_scale=CFG_SCALE
                )

                batch_mel_frames = x["mel_frames"]
                batch_silence_durations = x["next_mel_mute_duration_ms"]
                
                for i, pred in enumerate(pred_batch):
                    mel_frame = batch_mel_frames[i]
                    sliced_pred = pred[:, :mel_frame]
                    denormed_mel = inverse_zscore_normalize_with_uncond(
                        sliced_pred,
                        self.mel_stats["mean"],
                        self.mel_stats["std"],
                        TrainConfig.get_mel_uncond_token(),
                    )
                    mel_chunks.append(denormed_mel.cpu())
                
                for i, duration_tensor in enumerate(batch_silence_durations):
                    next_mel_duration_ms = int(duration_tensor.item())
                    if next_mel_duration_ms > 0:
                        silence_positions.append(len(mel_chunks) - len(batch_silence_durations) + i)
                        silence_chunks.append(next_mel_duration_ms)
        
        silence_mels = []
        if silence_chunks:
            for duration_ms in silence_chunks:
                silence_mel = create_silent_mel_spectrogram(duration_ms).squeeze().cpu()
                silence_mels.append(silence_mel)
        
        final_mels = []
        silence_idx = 0
        for i, mel_chunk in enumerate(mel_chunks):
            final_mels.append(mel_chunk)
            if silence_idx < len(silence_positions) and i == silence_positions[silence_idx]:
                final_mels.append(silence_mels[silence_idx])
                silence_idx += 1

        final_mel = torch.cat(final_mels, dim=1)
        logging.info("final mel shape: %s", tuple(final_mel.shape))
        audio = mel_spectrogram_to_audio(final_mel, device=self.device)
        logging.info("audio shape: %s", tuple(audio.shape))
        return audio

    def post_process(self, audio):
        ckpt_dir = os.path.dirname(self.checkpoint)
        out_dir = os.path.join(ckpt_dir, f"prediction_{get_time_str()}")
        os.makedirs(out_dir, exist_ok=True)

        wav_path = os.path.join(out_dir, f"audio_CFG_{CFG_SCALE}.wav")
        save_audio(audio, wav_path)
        logging.info("Saved audio at %s", wav_path)
        print(wav_path)

        shutil.copytree("./engine", os.path.join(out_dir, "_engine"))


if __name__ == "__main__":
    predictor = Predictor()
    audio = predictor.sample()
    predictor.post_process(audio)
