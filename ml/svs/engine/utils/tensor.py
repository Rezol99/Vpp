import torch


def verify_tensor(tensor: torch.Tensor, name: str = "Tensor") -> None:
    """テンソルにNaNやInfが含まれていないか検証する"""
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    print(f"{name} shape: {tensor.shape}")
    print(f"Has NaN? {has_nan}")
    print(f"Has Inf? {has_inf}")

    assert not has_nan, f"{name} has NaN values"
    assert not has_inf, f"{name} has Inf values"
