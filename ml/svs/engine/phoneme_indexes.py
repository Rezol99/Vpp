import json
from typing import ClassVar


class PhonemeIndexes:
    _PATH: ClassVar[str] = "./metadata/phoneme_indexes.json"
    _indexes: ClassVar[dict[str, int] | None] = None
    _reverse_indexes: ClassVar[dict[int, str] | None] = None

    @classmethod
    def _load_if_needed(cls) -> dict[str, int]:
        if cls._indexes is None:
            with open(cls._PATH, "r") as f:
                cls._indexes = json.load(f)["phonemes"]
        return cls._indexes  # type: ignore

    @classmethod
    def index(cls, phoneme: str) -> int:
        return cls._load_if_needed()[phoneme]

    @classmethod
    def phoneme(cls, index: int) -> str | None:
        if cls._reverse_indexes is None:
            cls._reverse_indexes = {v: k for k, v in cls._load_if_needed().items()}
        return cls._reverse_indexes.get(index)

    @classmethod
    def max_index(cls) -> int:
        return max(cls._load_if_needed().values())

    @property
    def pad_index(self) -> int:
        mute_index =  self._load_if_needed()["PAD"]
        return mute_index