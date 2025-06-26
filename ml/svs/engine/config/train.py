from dataclasses import dataclass
from typing import ClassVar

from engine.config.audio import AudioConfig
from engine.datasets_stats import DatasetStats


@dataclass
class TrainConfig:
    # インスタンス変数
    diffusion_steps: int = 100  # 拡散モデルのノイズのステップ数
    inference_steps: int = 12   # 推論時のステップ数（DPM-Solver++用）

    batch_size: int = 16
    epoch: int = 50

    lr: float = 1e-5
    eta_min: float = 1e-7
    num_workers: int = 4
    datasets_dir: str = "./datasets/*/DATABASE"
    mel_zero = 0.0
    phoneme_uncond_token: int = 0
    pad_value = -1  # パディング専用値（無条件トークンと分離）
    phoneme_emb_dim = 256
    
    # Z-score normalization unconditional value
    zscore_uncond_normalized_val: ClassVar[float] = -3.0  # より極端な値で明確に無条件を示す
    
    @classmethod
    def get_pitch_uncond_token(cls) -> float:
        """データセット統計から適切なpitch uncondトークンを計算"""
        pitch_stats = DatasetStats.get("pitches")
        # 有効範囲外の明確な値を使用（最小値より5半音低く）
        return float(pitch_stats["min"] - 5)
    
        
    @classmethod 
    def get_mel_uncond_token(cls) -> float:
        """Z-score正規化後の明確な無条件値を返す"""
        return cls.zscore_uncond_normalized_val

    max_phonemes_size: int = 6
    max_sequence_length: int = 512

    max_notes_size: int = 1024
    max_labs_size: int = 2048
    mask_labs_duration_value: float = -10.0
    mask_labs_phoneme_index: int = 0
    labs_stats_path = "./metadata/labs_stats.json"
    labs_phonemes_path = "./metadata/labs_phonemes.json"
    test_size: float = 0.05
    kl_weight = 0.0005
    kl_anneal_epochs = 10  # より早い収束でKL損失を活用

    part_mel_steps: int = (
        AudioConfig.audio.sampling_rate // AudioConfig.audio.hop_size
    )  # 1秒のメルスペクトラムのステップ数

    uncond_prob: float = 0.01  # 全体の無条件確率を削減

    uncond_lyric_prob: float = 0.05  # 個別の無条件確率を削減
    uncond_pitch_prob: float = 0.05
    uncond_prev_part_prob: float = 0.1
    free_bits: float = 0.1

    @classmethod
    def print_all(cls):
        print("\n=== All constants ===")
        for key, value in cls.__dict__.items():
            if not callable(value) and not key.startswith("__") and key != "print_all":
                print(f"{key}: {value}")
        print("=== END ====\n")


# TrainConfig.print_all()
