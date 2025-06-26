import json
import os
import platform
import sys
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from torch import Tensor

from engine.config.audio import AudioConfig
from engine.config.train import TrainConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '../hifi_gan'))

system = platform.system().lower()
if system == "linux":
    matplotlib.use("TkAgg")


class HiFiGANVocoder:
    """Direct HiFi-GAN inference wrapper to avoid subprocess overhead."""
    
    def __init__(self, checkpoint_path: str | None = None, config_path: str | None = None, device: torch.device | None = None):
        try:
            from env import AttrDict
            from models import Generator
            from meldataset import MAX_WAV_VALUE
        except ImportError:
            raise ImportError("Could not import HiFi-GAN modules. Make sure sys.path is set correctly.")
        
        self.MAX_WAV_VALUE = MAX_WAV_VALUE
        
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
            
        if checkpoint_path is None:
            checkpoint_path = "./engine/hifi_gan/g_02500000"
        if config_path is None:
            config_path = "./engine/hifi_gan/config.json"
            
        # Load config
        with open(config_path) as f:
            json_config = json.loads(f.read())
        self.h = AttrDict(json_config)
        
        # Initialize generator
        self.generator = Generator(self.h).to(self.device)
        
        # Load checkpoint
        assert os.path.isfile(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"
        checkpoint_dict = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint_dict["generator"])
        
        # Set to eval mode and remove weight norm
        self.generator.eval()
        self.generator.remove_weight_norm()
    
    def mel_to_audio(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to audio tensor."""
        with torch.no_grad():
            # Ensure mel is on the correct device and has the right shape
            if mel_spectrogram.device != self.device:
                mel_spectrogram = mel_spectrogram.to(self.device)
            
            # Add batch dimension if needed
            if mel_spectrogram.dim() == 2:
                mel_spectrogram = mel_spectrogram.unsqueeze(0)
            
            # Generate audio
            y_g_hat = self.generator(mel_spectrogram)
            audio = y_g_hat.squeeze()
            
            # Convert to proper range and format
            audio = audio * self.MAX_WAV_VALUE
            audio = audio.clamp(-self.MAX_WAV_VALUE, self.MAX_WAV_VALUE)
            
            # Ensure audio is 2D with shape [1, num_samples] for consistency
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Convert to float and normalize back
            audio = audio.float() / self.MAX_WAV_VALUE
            
            return audio.cpu()


# Global vocoder instances per device
_vocoder_instances = {}


def get_vocoder_instance(device: torch.device | None = None):
    """Get or create the HiFi-GAN vocoder instance for the specified device."""
    global _vocoder_instances
    if device is None:
        device = torch.device("cpu")
    
    device_str = str(device)
    if device_str not in _vocoder_instances:
        _vocoder_instances[device_str] = HiFiGANVocoder(device=device)
    return _vocoder_instances[device_str]


def resample_audio(audio: Tensor, sr: int, new_sr: int) -> Tensor:
    resampler = torchaudio.transforms.Resample(sr, new_sr)
    return resampler(audio)


def load_audio(wav: str) -> Tuple[Tensor, int]:
    audio, sr = torchaudio.load(wav)
    return audio, sr


def save_audio(
    waveform: Tensor, path: str, sample_rate=AudioConfig.audio.sampling_rate
):
    torchaudio.save(path, waveform, sample_rate)


def split_audio(
    waveform, sample_rate, segment_duration_ms, include_remainder=False
) -> list[Tensor]:
    segment_samples = int(sample_rate * (segment_duration_ms / 1000.0))

    if include_remainder:
        segments = [
            waveform[:, i : i + segment_samples]
            for i in range(0, waveform.size(1), segment_samples)
        ]
    else:
        segments = waveform.split(segment_samples, dim=1)

    return segments


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


# TODO: この実装で短いケースのpaddingをconstantに変えたがそのテストはしてないのでテストを行う
def tensor_to_mel_spectrogram(
    y: Tensor,
    sampling_rate=AudioConfig.audio.sampling_rate,
    n_fft=AudioConfig.audio.n_fft,
    n_mels=AudioConfig.audio.num_mels,
    f_min=AudioConfig.audio.fmin,
    hop_length=AudioConfig.audio.hop_size,
    win_size=AudioConfig.audio.win_size,
    center=False,
    f_max=AudioConfig.audio.fmax,
) -> Tensor:

    global mel_basis, hann_window
    key = f"{f_max}_{y.device}"
    if key not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max
        )
        mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    pad_len = (n_fft - hop_length) // 2
    y = y.unsqueeze(1)  # [B,1,T]

    # 「<= pad_len」で constant に切り替え
    if y.size(-1) <= pad_len:
        y = F.pad(
            y, (pad_len, pad_len), mode="constant", value=0.0
        )  # TODO: replicateでパディングするのを検討する
    else:
        y = F.pad(y, (pad_len, pad_len), mode="reflect")

    # それでも長さが n_fft 未満なら末尾にゼロ埋め
    if y.size(-1) < n_fft:
        extra = n_fft - y.size(-1)
        y = F.pad(
            y, (0, extra), mode="constant", value=0.0
        )  # TODO: replicateでパディングするのを検討する

    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.abs(spec) + 1e-9
    spec = torch.matmul(mel_basis[key], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_to_audio(mel_spectrogram: Tensor, cpu=False, device: torch.device | None = None):
    """Convert mel spectrogram to audio using direct HiFi-GAN inference.
    
    Args:
        mel_spectrogram: Input mel spectrogram tensor
        cpu: Whether to force CPU inference (kept for compatibility)
        device: Device to use for inference. If None, auto-detect from mel_spectrogram
    
    Returns:
        Audio tensor
    """
    # Determine device
    if cpu:
        target_device = torch.device("cpu")
    elif device is not None:
        target_device = device
    else:
        # Auto-detect from input tensor
        target_device = mel_spectrogram.device
    
    # Use the vocoder instance for the specified device
    vocoder = get_vocoder_instance(target_device)
    return vocoder.mel_to_audio(mel_spectrogram)


def plot_mel_spectrogram(mel_spectrogram: Tensor, title="Mel Spectrogram"):
    plt.figure(figsize=(10, 4))

    if mel_spectrogram.device.type != "cpu":
        mel_spectrogram = mel_spectrogram.cpu()

    mel_spectrogram_db = torchaudio.functional.amplitude_to_DB(
        mel_spectrogram, multiplier=10.0, amin=1e-10, db_multiplier=0.0
    )
    plt.imshow(
        mel_spectrogram_db.numpy(), aspect="auto", origin="lower", cmap="viridis"
    )
    plt.title(title)
    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (mel)")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.show()


def normalize_y(tensor, json_path="./data/encode_params.json", min_y=None, max_y=None):
    """
    テンソルを [-1, 1] の範囲に正規化する。
    """
    if min_y is None:
        min_y = tensor.min().item()
    if max_y is None:
        max_y = tensor.max().item()

    tensor_normalized = 2 * (tensor - min_y) / (max_y - min_y) - 1  # [-1, 1] に正規化

    with open(json_path, "w") as f:
        json.dump(
            {
                "min_y": min_y,
                "max_y": max_y,
            },
            f,
            indent=4,
        )

    return tensor_normalized


def slice_mel_spectrogram(
    mel_spec: torch.Tensor,
    start_ms: int | float,
    end_ms: int | float,
    sr=AudioConfig.audio.sampling_rate,
    hop_length=AudioConfig.audio.hop_size,
) -> torch.Tensor:
    # 1フレームの時間 (ms)
    frame_duration_ms = (hop_length / sr) * 1000  # hop_lengthサンプルが何msか

    # ミリ秒をフレームに変換
    start_frame = int(start_ms / frame_duration_ms)
    end_frame = int(end_ms / frame_duration_ms)

    # スライス
    return mel_spec[:, start_frame:end_frame]


def slice_audio(
    audio: torch.Tensor,
    start_ms: float,
    end_ms: float,
    sr=AudioConfig.audio.sampling_rate,
) -> torch.Tensor:
    """
    ミリ秒単位でオーディオをスライス
    :param audio: (Tensor) [1, num_samples] 形式のオーディオ
    :param sr: (int) サンプリングレート
    :param start_ms: (int) スライスの開始時間（ミリ秒）
    :param end_ms: (int) スライスの終了時間（ミリ秒）
    :return: スライスされたオーディオ
    """
    # ミリ秒をサンプル数に変換
    start_sample = int((start_ms / 1000) * sr)
    end_sample = int((end_ms / 1000) * sr)

    # スライスを適用
    return audio[:, start_sample:end_sample]


def visualize_mask_noise(
    mask, noise, noisy, target, index=0, log_scale=False, figsize=(16, 4)
):
    """
    mask, noise, noisy, target はいずれも (B, F, T) などの形状を想定
    index: バッチ内で可視化したいサンプルのインデックス
    log_scale: True にすると log(1 + x) でスケーリングして表示 (スペクトログラム表示用)
    figsize: プロット全体のサイズ (横, 縦)
    """

    # バッチの index 番目を取り出して CPU へ移動、numpy 配列に変換
    mask_sample = mask[index].detach().cpu().numpy().T
    noise_sample = noise[index].detach().cpu().numpy().T
    noisy_sample = noisy[index].detach().cpu().numpy().T
    target_sample = target[index].detach().cpu().numpy().T

    # ログスケール変換 (スペクトログラムとして見やすくするため)
    if log_scale:
        # clamp で負の値を避けつつ、log(1 + x) をとるなど、スケールを調整
        # 以下は例なので必要に応じて実装を変更してください
        noise_sample = np.log1p(np.abs(noise_sample))
        noisy_sample = np.log1p(np.abs(noisy_sample))
        target_sample = np.log1p(np.abs(target_sample))
        # mask は 0/1 などのバイナリの場合、ログスケールは必要ないかもしれません

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # 1. マスク
    im0 = axes[0].imshow(mask_sample, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Mask")
    fig.colorbar(im0, ax=axes[0])

    # 2. 元のターゲット
    im1 = axes[1].imshow(target_sample, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_title("Target Spectrogram")
    fig.colorbar(im1, ax=axes[1])

    # 3. 加えたノイズ
    im2 = axes[2].imshow(noise_sample, aspect="auto", origin="lower", cmap="viridis")
    axes[2].set_title("Added Noise")
    fig.colorbar(im2, ax=axes[2])

    # 4. ノイズを加えた後のスペクトログラム
    im3 = axes[3].imshow(noisy_sample, aspect="auto", origin="lower", cmap="viridis")
    axes[3].set_title("Noisy Spectrogram")
    fig.colorbar(im3, ax=axes[3])

    plt.tight_layout()
    plt.show()


def create_silent_audio(
    duration_ms: float, sr=AudioConfig.audio.sampling_rate
) -> torch.Tensor:
    """
    指定された時間の無音オーディオを生成します。

    Args:
        duration_ms (float): 生成するオーディオの長さ（ミリ秒）
        sr (int, optional): サンプリングレート。デフォルトはAudioConfigの設定値

    Returns:
        torch.Tensor: [1, サンプル数] の形状を持つ無音オーディオテンソル
    """
    # ミリ秒をサンプル数に変換
    num_samples = int((duration_ms / 1000) * sr)

    # ゼロテンソルを生成（無音）
    silence = torch.zeros(1, num_samples)

    return silence

def create_silent_mel_spectrogram(
    duration_ms: float,
    sr: int = AudioConfig.audio.sampling_rate,
    n_fft = AudioConfig.audio.n_fft,
    n_mels = AudioConfig.audio.num_mels,
    f_min = AudioConfig.audio.fmin,
    f_max = AudioConfig.audio.fmax,
    hop_length: int = AudioConfig.audio.hop_size,
    win_size: int = AudioConfig.audio.win_size,
    center: bool = False,
) -> Tensor:
    """
    duration_msミリ秒の無音オーディオからMelスペクトログラムを生成して返す
    """
    silent_audio = create_silent_audio(duration_ms, sr)
    mel = tensor_to_mel_spectrogram(
        silent_audio,
        sampling_rate=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        hop_length=hop_length,
        win_size=win_size,
        center=center,
    )
    return mel