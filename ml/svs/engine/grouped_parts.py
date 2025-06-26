import glob
import json
from typing import cast, Union, Literal

from engine.types.grouped_part import GroupedPhonemePart, GroupedMutePart, Metadata

class GroupedParts:
    def __init__(self, prediction: bool = False):
        self._all_matadata: list[Metadata] = []

        self.prediction = prediction
        self._load_if_needed()

    def _glob_metadata_paths(self) -> list[str]:
        pattern = (
            "./datasets/**/grouped_duration/*/metadata.json"
            if not self.prediction
            else "./datasets/_TEST/**/grouped_duration/*/metadata.json"
        )
        paths = sorted(glob.glob(pattern, recursive=True))
        assert paths, f"No grouped metadata files found with pattern {pattern}"
        return paths

    @staticmethod 
    def _map_type_part(part: dict | None):
        if part is None: return None

        part_type = part.get("type")
        base_kwargs = {k: v for k, v in part.items() if k != "type"}
        if part_type == "PHONEME":
            return GroupedPhonemePart(**base_kwargs)
        elif part_type == "MUTE":
            return GroupedMutePart(**base_kwargs)

    @staticmethod
    def _load_metadata(path: str) -> Metadata:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        current_part = GroupedParts._map_type_part(data["current"])
        prev_part = GroupedParts._map_type_part(data["prev"])
        next_part = GroupedParts._map_type_part(data["next"])

        return Metadata(current=current_part, prev=prev_part, next=next_part)

    def _load_if_needed(self) -> list[Metadata]:
        if not self._all_matadata:
            metadata_paths = self._glob_metadata_paths()
            self._all_matadata = [
                self._load_metadata(p) for p in metadata_paths
            ]
        return self._all_matadata

    def get_all_metadata(self) -> list[Metadata]:
        return self._load_if_needed()
