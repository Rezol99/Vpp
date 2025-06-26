from datetime import datetime

import pytz


def get_time_str():
    return datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%Y-%m-%d_%H:%M:%S")


def hts_time_to_ms(hts_time: float | int):
    return hts_time / 10_000


def ms_to_hts_time(ms: float | int) -> int:
    return int(ms * 10_000)
