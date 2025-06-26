import json
import os

import numpy as np
import torch

from engine.types.torch_like import TensorLike


def compute_statistics(values, save_path=None):
    """
    values: 1次元のNumPy配列またはリスト
    save_path: 統計量を保存するファイルパス（例: "metadata/dataset_stats.json"）
    return: stats = {"mean": ..., "std": ...}
    """
    values = np.array(values)
    mean = float(values.mean())
    std = float(values.std())
    stats = {"mean": mean, "std": std}
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)
    return stats


def load_statistics(load_path):
    """保存した統計量を読み込む"""
    with open(load_path, "r") as f:
        stats = json.load(f)
    return stats


def normalize(x: TensorLike, mean: TensorLike, std: TensorLike) -> torch.Tensor:
    """Zスコア正規化: (x - mean) / std"""
    x = torch.as_tensor(x, dtype=torch.float32)
    mean = torch.as_tensor(mean, dtype=torch.float32)
    std = torch.as_tensor(std, dtype=torch.float32)
    return (x - mean) / std


def inverse_normalize(x_norm, mean, std):
    """逆正規化: x_norm * std + mean"""
    return x_norm * std + mean


def min_max_normalize(x: torch.Tensor, min_val, max_val) -> torch.Tensor:
    """Min-Max 正規化: (x - min) / (max - min)"""
    return (x - min_val) / (max_val - min_val)


def inverse_min_max_normalize(x_norm: torch.Tensor, min_val, max_val) -> torch.Tensor:
    """逆 Min-Max 正規化: x_norm * (max - min) + min"""
    return x_norm * (max_val - min_val) + min_val


def zscore_normalize_with_uncond(x: torch.Tensor, mean: TensorLike, std: TensorLike, uncond_val, uncond_normalized_val: float = None) -> torch.Tensor:
    """
    無条件対応Z-score正規化: 無条件を指定値、それ以外をZ-score正規化
    - uncond_val と一致する値は uncond_normalized_val にマップ
    - それ以外は標準的なZ-score正規化 (x - mean) / std
    """
    if uncond_normalized_val is None:
        from engine.config.train import TrainConfig
        uncond_normalized_val = TrainConfig.zscore_uncond_normalized_val
        
    x = torch.as_tensor(x, dtype=torch.float32)
    mean = torch.as_tensor(mean, dtype=torch.float32)
    std = torch.as_tensor(std, dtype=torch.float32)
    
    is_uncond = torch.isclose(x, torch.tensor(uncond_val, dtype=x.dtype, device=x.device))
    
    # 通常の値をZ-score正規化
    normalized = (x - mean) / std
    
    # 無条件値を指定値に設定
    result = torch.where(is_uncond, torch.full_like(x, uncond_normalized_val), normalized)
    return result


def inverse_zscore_normalize_with_uncond(x_norm: torch.Tensor, mean: TensorLike, std: TensorLike, uncond_val, uncond_normalized_val: float = None) -> torch.Tensor:
    """
    無条件対応逆Z-score正規化
    - uncond_normalized_val の値は uncond_val にマップ
    - それ以外は標準的な逆Z-score正規化 x_norm * std + mean
    """
    if uncond_normalized_val is None:
        from engine.config.train import TrainConfig
        uncond_normalized_val = TrainConfig.zscore_uncond_normalized_val
        
    mean = torch.as_tensor(mean, dtype=torch.float32)
    std = torch.as_tensor(std, dtype=torch.float32)
    
    is_uncond = torch.isclose(x_norm, torch.tensor(uncond_normalized_val, dtype=x_norm.dtype, device=x_norm.device))
    
    # 通常の値を逆Z-score正規化
    denormalized = x_norm * std + mean
    
    # 無条件値を設定
    result = torch.where(is_uncond, torch.full_like(x_norm, uncond_val), denormalized)
    return result


def min_max_normalize_with_uncond(x: torch.Tensor, min_val, max_val, uncond_val) -> torch.Tensor:
    """
    無条件対応Min-Max正規化: 無条件を0、それ以外を0.1-1の範囲に正規化
    - uncond_val と一致する値は 0 にマップ
    - それ以外は 0.1-1 の範囲に線形マップ
    """
    is_uncond = torch.isclose(x, torch.tensor(uncond_val, dtype=x.dtype, device=x.device))
    
    # 通常の値を0.1-1範囲に正規化
    normalized = (x - min_val) / (max_val - min_val) * 0.9 + 0.1
    
    # 無条件値を0に設定
    result = torch.where(is_uncond, torch.zeros_like(x), normalized)
    return result


def inverse_min_max_normalize_with_uncond(x_norm: torch.Tensor, min_val, max_val, uncond_val) -> torch.Tensor:
    """
    無条件対応逆Min-Max正規化
    - 0 の値は uncond_val にマップ
    - 0.1-1 の値は元の範囲に逆変換
    """
    is_uncond = torch.isclose(x_norm, torch.zeros_like(x_norm))
    
    # 0.1-1範囲から元の範囲に逆変換
    denormalized = (x_norm - 0.1) / 0.9 * (max_val - min_val) + min_val
    
    # 無条件値を設定
    result = torch.where(is_uncond, torch.full_like(x_norm, uncond_val), denormalized)
    return result
