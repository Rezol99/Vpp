def pad_list(lst: list, target_length: int, pad_value: int | float | str | None = 0):
    return lst + [pad_value] * (target_length - len(lst))
