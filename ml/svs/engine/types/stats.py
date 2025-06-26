from typing import TypedDict

import torch


class Stats(TypedDict):
    mean: torch.Tensor | int | float
    std: torch.Tensor | int | float
    min: torch.Tensor | int | float
    max: torch.Tensor | int | float
