import json
from typing import ClassVar, Dict, Literal

from engine.types.stats import Stats


class DatasetStats:
    _PATH: ClassVar[str] = "metadata/dataset_stats.json"
    _stats: ClassVar[Dict[str, Stats] | None] = None

    @classmethod
    def _load_if_needed(cls) -> Dict[str, Stats]:
        if cls._stats is None:
            with open(cls._PATH, "r") as f:
                raw_stats = json.load(f)["stats"]
                cls._stats = {k: Stats(**v) for k, v in raw_stats.items()}
        return cls._stats

    @classmethod
    def get(cls, type: Literal["duration_ms", "pitches", "mel", "sequence_length"]) -> Stats:
        return cls._load_if_needed()[type]
