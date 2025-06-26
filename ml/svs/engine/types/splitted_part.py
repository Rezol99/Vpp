from dataclasses import dataclass
from typing import Optional

from engine.types.note_segment import MuteSegment, NoteSegment, PhonemeSegment


@dataclass
class BaseSplittedPart:
    index: int
    audio_part_file: str
    prev_audio_part_file: Optional[str]


@dataclass
class SplittedPhonemePart(BaseSplittedPart):
    segment: PhonemeSegment


@dataclass
class SplittedMutePart(BaseSplittedPart):
    segment: MuteSegment


@dataclass
class SplittedPart(BaseSplittedPart):
    segment: NoteSegment
