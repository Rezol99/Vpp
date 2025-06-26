import json
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
from glob import glob
from os.path import basename, dirname, join, splitext

from tqdm.auto import tqdm

from engine.config.project import ProjectConfig
from engine.utaupy.hts import CustomNotes
from engine.utaupy.utils._ust2hts import ust2hts

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
MUTE_SET = {"sil", "sli", "pau", "xx"}

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------


def _build_pitch_context(notes):
    """Return a list[dict] `pitch_contexts` aligned with *notes* indices.

    Each dict has keys: prev2, prev1, current, next1, next2, next3.
    Only **non‑mute** notes are considered for the context; mute notes are skipped.
    If a neighbour is absent, None is used (→ null in JSON).
    """
    total = len(notes)
    pitch_ctx_list: list = [None] * total

    # Collect indices of non‑mute notes
    non_mute_idx = [
        i for i, n in enumerate(notes) if n.phoneme_context.current not in MUTE_SET
    ]

    # Build mapping idx → position in non_mute_idx list for O(1) neighbour lookup
    idx2pos = {idx: pos for pos, idx in enumerate(non_mute_idx)}

    for idx in non_mute_idx:
        pos = idx2pos[idx]
        get_idx = lambda offset: (
            non_mute_idx[pos + offset]
            if 0 <= pos + offset < len(non_mute_idx)
            else None
        )

        prev2_idx = get_idx(-2)
        prev1_idx = get_idx(-1)
        next1_idx = get_idx(1)
        next2_idx = get_idx(2)
        next3_idx = get_idx(3)

        pitch_ctx_list[idx] = {
            "prev2": notes[prev2_idx].notenum if prev2_idx is not None else None,
            "prev1": notes[prev1_idx].notenum if prev1_idx is not None else None,
            "current": notes[idx].notenum,
            "next1": notes[next1_idx].notenum if next1_idx is not None else None,
            "next2": notes[next2_idx].notenum if next2_idx is not None else None,
            "next3": notes[next3_idx].notenum if next3_idx is not None else None,
        }

    return pitch_ctx_list


def notes_to_segments(notes, ust_name: str, mute_set=MUTE_SET):
    """Convert a list of CustomNote → list of OrderedDict segments.

    * Continuous mute notes are merged into a single `MUTE` segment.
    * Each voiced note becomes a `PHONEME` segment that now contains **pitch_context**
      mirroring the structure of *phoneme_context*.
    """
    segments = []
    seg_start = seg_end = None

    # Pre‑compute pitch context for every note (None for mute indices)
    pitch_ctx_arr = _build_pitch_context(notes)

    def push_mute_segment():
        nonlocal seg_start, seg_end
        if seg_start is None:
            return
        seg = OrderedDict()
        seg["type"] = "MUTE"
        seg["start_time"] = seg_start
        seg["end_time"] = seg_end
        segments.append(seg)
        seg_start = seg_end = None

    for idx, note in enumerate(notes):
        is_mute = note.phoneme_context.current in mute_set

        if is_mute:
            # Collect consecutive mute range
            if seg_start is None:
                seg_start = note.start_time
            seg_end = note.end_time
            continue

        # Flush previous mute range (if any)
        push_mute_segment()

        # Build PHONEME segment with full contexts
        note_dict = asdict(deepcopy(note))
        ordered = OrderedDict()
        ordered["type"] = "PHONEME"
        ordered.update(note_dict)

        # Attach pitch_context (already computed)
        ordered["pitch_context"] = pitch_ctx_arr[idx]

        segments.append(ordered)

    # Flush trailing mute range
    push_mute_segment()

    # Assign deterministic id & file path
    for i, seg in enumerate(segments):
        seg["id"] = join(dirname(ust_name), "parts", f"{i:03d}")
        seg["file"] = join(seg["id"], "part.wav")

    return segments


# ------------------------------------------------------------
# Main script
# ------------------------------------------------------------
if __name__ == "__main__":
    # Collect *.ust files recursively
    files = sorted(glob(join(ProjectConfig.db_root, "**/*.ust"), recursive=True))

    for ust_path in tqdm(files, "Converting ust to segments"):
        ust_name = splitext(basename(ust_path))[0]

        # Convert UST → CustomNotes
        ust_custom_notes = ust2hts(
            ust_path,
            None,
            ProjectConfig.table_path,
            strict_sinsy_style=False,
            as_mono=False,
            custom=True,
        )
        if not isinstance(ust_custom_notes, CustomNotes):
            raise ValueError("Invalid type !")

        # --------------------------------------------------
        # ideal segments.json (without natural timing)
        # --------------------------------------------------
        segments = notes_to_segments(ust_custom_notes.notes, ust_path)
        output_path = join(dirname(ust_path), f"{ust_name}_segments.json")
        with open(output_path, "w") as f:
            json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)

        # --------------------------------------------------
        # natural_segments.json (using lab timing)
        # --------------------------------------------------
        natural_segments = []
        natural_lab_path = join(dirname(ust_path), f"{ust_name}_grouped_mute.lab")

        with open(natural_lab_path, "r") as f:
            lab_lines = f.readlines()

        for i, line in enumerate(lab_lines):
            start_time, end_time, phoneme = line.strip().split(" ")
            # Skip terminal MUTE line to avoid IndexError
            if i == len(lab_lines) - 1 and phoneme == "MUTE":
                break
            seg = segments[i]
            if seg["type"] == "PHONEME":
                assert seg["phoneme_context"]["current"] == phoneme
            seg["start_time"] = int(start_time)
            seg["end_time"] = int(end_time)
            natural_segments.append(seg)

        natural_output_path = join(
            dirname(ust_path), f"{ust_name}_natural_segments.json"
        )
        with open(natural_output_path, "w") as f:
            json.dump({"segments": natural_segments}, f, ensure_ascii=False, indent=2)
