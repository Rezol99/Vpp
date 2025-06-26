import math

def next_power_of_two_greater_than(x: float) -> int:
    """
    xより大きい最小の2のべき乗を返す（x自身は含まない）
    """
    return 2 ** math.ceil(math.log2(x))