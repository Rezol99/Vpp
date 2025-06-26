import random

import numpy as np
import torch


def set_seed(seed=3407):
    """
    Pythonの各種ライブラリのシードを固定する関数

    Args:
        seed (int): 固定するシード値。デフォルトは3407
    """

    # Pythonのrandomシードを固定
    random.seed(seed)

    # NumPyのシードを固定
    np.random.seed(seed)

    # PyTorchのシードを固定（CPU用）
    torch.manual_seed(seed)

    # PyTorchのシードを固定（GPU用）
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # マルチGPUの場合

    # パフォーマンスに影響する可能性がある設定はコメントアウト
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
