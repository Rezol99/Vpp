import glob
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import torch
from tqdm.auto import tqdm

from engine.config.audio import AudioConfig
from engine.types.note_segment import PhonemeSegment
from engine.utils.time import hts_time_to_ms
from engine.utils.audio import (
    load_audio,
    resample_audio,
    slice_audio,
    save_audio,
    tensor_to_mel_spectrogram,
)
import logging


DIR_NAME = "grouped_duration"
MAX_DURATION = 4000  # ms


class Stats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, x: float):
        # Welford's online algorithm
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.min = min(self.min, x)
        self.max = max(self.max, x)

    def finalize(self):
        std = (self.M2 / (self.count - 1)) ** 0.5 if self.count > 1 else 0.0
        return {"mean": self.mean, "std": std, "min": self.min, "max": self.max}


def merge_stats(stats1: Stats, stats2: Stats) -> Stats:
    """
    Merge two Stats objects using Welford's algorithm for combining statistics
    """
    if stats1.count == 0:
        return stats2
    if stats2.count == 0:
        return stats1
    
    merged = Stats()
    merged.count = stats1.count + stats2.count
    
    # Combine means
    delta = stats2.mean - stats1.mean
    merged.mean = stats1.mean + delta * stats2.count / merged.count
    
    # Combine M2 (for variance calculation)
    merged.M2 = stats1.M2 + stats2.M2 + delta * delta * stats1.count * stats2.count / merged.count
    
    # Combine min/max
    merged.min = min(stats1.min, stats2.min)
    merged.max = max(stats1.max, stats2.max)
    
    return merged


def process_segment_group(max_sec_segments, audio_path, idx):
    grp_id = str(idx).zfill(3)
    out_dir = os.path.join(os.path.dirname(audio_path), DIR_NAME, grp_id)
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "type": "PHONEME",
        "id": out_dir,
        "duration_ms": 0.0,
    }
    for seg in max_sec_segments:
        d = hts_time_to_ms(seg.end_time - seg.start_time)
        meta["duration_ms"] += d

    audio, sr = load_audio(audio_path)
    if sr != AudioConfig.audio.sampling_rate:
        audio = resample_audio(audio, sr, AudioConfig.audio.sampling_rate)
    start_ms = hts_time_to_ms(max_sec_segments[0].start_time)
    sliced = slice_audio(audio, start_ms, start_ms + meta["duration_ms"])
    save_audio(sliced, os.path.join(out_dir, "audio.wav"))
    mel = tensor_to_mel_spectrogram(sliced)
    mel_path = os.path.join(out_dir, "mel.pt")
    torch.save(mel, mel_path)
    meta["mel_path"] = mel_path

    # フレームレベルで条件を展開
    mel_frames = mel.shape[-1]  # melスペクトログラムのフレーム数
    phonemes = []
    pitches = []
    
    # 各音素の継続時間をフレーム数に変換
    total_duration_ms = meta["duration_ms"]
    ms_per_frame = total_duration_ms / mel_frames if mel_frames > 0 else 0
    
    frame_idx = 0
    for i, seg in enumerate(max_sec_segments):
        phoneme = seg.phoneme_context["current"]
        duration_ms = hts_time_to_ms(seg.end_time - seg.start_time)
        # Convert note number to Hz using MIDI note formula: freq = 440 * 2^((note-69)/12)
        pitch_hz = 440.0 * (2.0 ** ((seg.notenum - 69) / 12.0)) if seg.notenum > 0 else 0.0
        
        frames_for_phoneme = max(1, int(duration_ms / ms_per_frame)) if ms_per_frame > 0 else 1
        
        # 最後の音素の場合、残りのフレームをすべて割り当て
        if i == len(max_sec_segments) - 1:
            frames_for_phoneme = mel_frames - frame_idx
        
        # 各フレームに音素とピッチを割り当て
        for _ in range(frames_for_phoneme):
            if frame_idx < mel_frames:
                phonemes.append(phoneme)
                pitches.append(pitch_hz)
                frame_idx += 1
    
    # フレーム数が不足している場合は最後の値で埋める
    while len(phonemes) < mel_frames:
        if phonemes:
            phonemes.append(phonemes[-1])
            pitches.append(pitches[-1])
        else:
            phonemes.append("sil")
            pitches.append(0)
    
    # フレーム数が超過している場合は切り詰める
    phonemes = phonemes[:mel_frames]
    pitches = pitches[:mel_frames]
    
    # フレームレベルの条件をメタデータに追加
    meta["phonemes"] = phonemes
    meta["pitches"] = pitches
    meta["mel_frames"] = mel_frames

    return meta, sliced


def process_single_file(seg_path):
    """Process single segment file"""
    stats_data = {
        'duration_stats': Stats(),
        'pitch_stats': Stats(),
        'mel_stats': Stats(),
        'length_stats': Stats(),
        'phoneme_durations': [],
        'unique_phonemes': set()
    }
    
    base_dir = os.path.dirname(seg_path)
    clean_dir = os.path.join(base_dir, DIR_NAME)
    if os.path.exists(clean_dir):
        shutil.rmtree(clean_dir)

    with open(seg_path, "r", encoding="utf-8") as f:
        seg_data = json.load(f)

    audio_path = sorted(glob.glob(os.path.join(os.path.dirname(seg_path), "*.wav")))[0]
    max_sec_segments = []
    all_groups = []
    seg_duration_ms = 0.0

    for seg in seg_data["segments"]:
        if seg["type"] == "PHONEME":
            raw_ps = PhonemeSegment(**seg)
            raw_d_ms = hts_time_to_ms(raw_ps.end_time - raw_ps.start_time)

            chunks = []
            if raw_d_ms <= MAX_DURATION:
                chunks.append((raw_ps, raw_d_ms))
            else:
                phoneme = raw_ps.phoneme_context["current"]
                seg_start = raw_ps.start_time
                chunk_hts = MAX_DURATION * 10_000 / 1000.0
                full_chunks = int(raw_d_ms // MAX_DURATION)
                remainder = raw_d_ms % MAX_DURATION
                
                for i in range(full_chunks):
                    seg_dict = {**seg}
                    chunk_start = seg_start + i * chunk_hts
                    chunk_end = seg_start + (i + 1) * chunk_hts
                    seg_dict["start_time"] = chunk_start
                    seg_dict["end_time"] = chunk_end
                    ps_chunk = PhonemeSegment(**seg_dict)
                    actual_duration = hts_time_to_ms(chunk_end - chunk_start)
                    chunks.append((ps_chunk, actual_duration))
                
                if remainder > 0:
                    seg_dict = {**seg}
                    remainder_start = seg_start + full_chunks * chunk_hts
                    remainder_hts = remainder * 10_000 / 1000.0
                    remainder_end = remainder_start + remainder_hts
                    seg_dict["start_time"] = remainder_start
                    seg_dict["end_time"] = remainder_end
                    ps_chunk = PhonemeSegment(**seg_dict)
                    actual_remainder = hts_time_to_ms(remainder_end - remainder_start)
                    chunks.append((ps_chunk, actual_remainder))

            for i, (ps_chunk, d_ms) in enumerate(chunks):
                phoneme = ps_chunk.phoneme_context["current"]
                stats_data['phoneme_durations'].append((phoneme, d_ms))
                stats_data['unique_phonemes'].add(phoneme)
                
                # 現在のセグメントを追加すると4000msを超える場合の処理
                if seg_duration_ms + d_ms > MAX_DURATION and max_sec_segments:
                    # 現在のグループを4000msぴったりで分割
                    remaining_time = MAX_DURATION - seg_duration_ms
                    
                    if remaining_time > 0:
                        # 現在のセグメントを分割: 一部を現在のグループに追加
                        seg_dict = ps_chunk.__dict__.copy()
                        # 時間を計算（HTSユニット）
                        original_duration_hts = ps_chunk.end_time - ps_chunk.start_time
                        remaining_ratio = remaining_time / d_ms
                        split_duration_hts = int(original_duration_hts * remaining_ratio)
                        
                        # 前半部分（現在のグループに追加）
                        seg_dict["end_time"] = ps_chunk.start_time + split_duration_hts
                        first_part = PhonemeSegment(**seg_dict)
                        max_sec_segments.append(first_part)
                        
                        # 現在のグループを処理
                        meta, audio_slice = process_segment_group(
                            max_sec_segments, audio_path, len(all_groups)
                        )
                        all_groups.append(meta)
                        stats_data['duration_stats'].update(meta["duration_ms"])
                        
                        # Vectorized pitch stats update
                        if meta["pitches"]:
                            pitch_array = meta["pitches"]
                            for p in pitch_array:
                                stats_data['pitch_stats'].update(p)
                            stats_data['length_stats'].update(len(pitch_array))
                        
                        # Vectorized MEL stats update
                        mel = tensor_to_mel_spectrogram(audio_slice)
                        mel_values = mel.detach().cpu().numpy().ravel() if hasattr(mel, "detach") else mel.ravel()
                        for v in mel_values:
                            stats_data['mel_stats'].update(float(v))
                        
                        # 後半部分（次のグループの開始）
                        seg_dict["start_time"] = ps_chunk.start_time + split_duration_hts
                        seg_dict["end_time"] = ps_chunk.end_time
                        second_part = PhonemeSegment(**seg_dict)
                        remaining_duration = d_ms - remaining_time
                        
                        # 新しいグループを開始
                        max_sec_segments = [second_part]
                        seg_duration_ms = remaining_duration
                    else:
                        # 現在のグループを処理（分割なし）
                        meta, audio_slice = process_segment_group(
                            max_sec_segments, audio_path, len(all_groups)
                        )
                        all_groups.append(meta)
                        stats_data['duration_stats'].update(meta["duration_ms"])
                        
                        # Vectorized pitch stats update
                        if meta["pitches"]:
                            pitch_array = meta["pitches"]
                            for p in pitch_array:
                                stats_data['pitch_stats'].update(p)
                            stats_data['length_stats'].update(len(pitch_array))
                        
                        # Vectorized MEL stats update
                        mel = tensor_to_mel_spectrogram(audio_slice)
                        mel_values = mel.detach().cpu().numpy().ravel() if hasattr(mel, "detach") else mel.ravel()
                        for v in mel_values:
                            stats_data['mel_stats'].update(float(v))
                        
                        # 新しいグループを開始
                        max_sec_segments = [ps_chunk]
                        seg_duration_ms = d_ms
                else:
                    # 4000msを超えない場合は普通に追加
                    max_sec_segments.append(ps_chunk)
                    seg_duration_ms += d_ms

        elif seg["type"] == "MUTE":
            if max_sec_segments:
                meta, audio_slice = process_segment_group(
                    max_sec_segments, audio_path, len(all_groups)
                )
                if meta["duration_ms"] > MAX_DURATION:
                    print(f"WARNING: Group duration {meta['duration_ms']}ms exceeds {MAX_DURATION}ms!")
                    import sys
                    sys.exit(1)
                all_groups.append(meta)
                stats_data['duration_stats'].update(meta["duration_ms"])
                
                # Vectorized pitch stats update
                if meta["pitches"]:
                    pitch_array = meta["pitches"]
                    for p in pitch_array:
                        stats_data['pitch_stats'].update(p)
                    stats_data['length_stats'].update(len(pitch_array))
                
                # Vectorized MEL stats update
                mel = tensor_to_mel_spectrogram(audio_slice)
                mel_values = mel.detach().cpu().numpy().ravel() if hasattr(mel, "detach") else mel.ravel()
                for v in mel_values:
                    stats_data['mel_stats'].update(float(v))

            d_ms = hts_time_to_ms(seg["end_time"] - seg["start_time"])
            grp_id = str(len(all_groups)).zfill(3)
            out_dir = os.path.join(os.path.dirname(seg_path), DIR_NAME, grp_id)
            os.makedirs(out_dir, exist_ok=True)
            mute_meta = {
                "type": "MUTE",
                "id": out_dir,
                "duration_ms": d_ms,
            }
            with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(mute_meta, f, indent=2, ensure_ascii=False)
            all_groups.append(mute_meta)
            max_sec_segments = []
            seg_duration_ms = 0.0

    if max_sec_segments:
        meta, audio_slice = process_segment_group(
            max_sec_segments, audio_path, len(all_groups)
        )
        if meta["duration_ms"] > MAX_DURATION:
            print(f"WARNING: Group duration {meta['duration_ms']}ms exceeds {MAX_DURATION}ms!")
            import sys
            sys.exit(1)
        all_groups.append(meta)
        stats_data['duration_stats'].update(meta["duration_ms"])
        
        # Vectorized pitch stats update
        if meta["pitches"]:
            pitch_array = meta["pitches"]
            for p in pitch_array:
                stats_data['pitch_stats'].update(p)
            stats_data['length_stats'].update(len(pitch_array))
        
        # Vectorized MEL stats update
        mel = tensor_to_mel_spectrogram(audio_slice)
        mel_values = mel.detach().cpu().numpy().ravel() if hasattr(mel, "detach") else mel.ravel()
        for v in mel_values:
            stats_data['mel_stats'].update(float(v))

    # Add context information
    for i, group in enumerate(all_groups):
        current = dict(group)
        prev = dict(all_groups[i-1]) if i > 0 else None
        next_group_data = dict(all_groups[i+1]) if i < len(all_groups) - 1 else None
        
        updated_meta = {
            "current": current,
            "prev": prev,
            "next": next_group_data
        }
        
        meta_path = os.path.join(group["id"], "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(updated_meta, f, indent=2, ensure_ascii=False)

    # per-song metadata.json
    out_json = os.path.join(os.path.dirname(seg_path), DIR_NAME, "metadata.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_groups, f, indent=2, ensure_ascii=False)
    
    return stats_data


def main():
    import sys
    
    segment_paths = sorted(glob.glob("./datasets/*/DATABASE/*/*_natural_segments.json"))
    
    # Early quit for testing - process only first file
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        segment_paths = segment_paths[:1]
    
    # Initialize combined stats
    duration_stats = Stats()
    pitch_stats = Stats()
    mel_stats = Stats()
    length_stats = Stats()
    phoneme_durations = []
    unique_phonemes = set()
    
    # Use multiprocessing for parallel processing
    max_workers = min(cpu_count(), len(segment_paths), 4)  # Limit to 4 to avoid memory issues
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(process_single_file, path): path for path in segment_paths}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_path), total=len(segment_paths), desc="Processing files"):
            try:
                stats_data = future.result()
                
                # Merge stats from each file  
                phoneme_durations.extend(stats_data['phoneme_durations'])
                unique_phonemes.update(stats_data['unique_phonemes'])
                
                # Merge stats properly by combining Stats objects
                file_duration_stats = stats_data['duration_stats']
                file_pitch_stats = stats_data['pitch_stats']
                file_mel_stats = stats_data['mel_stats']
                file_length_stats = stats_data['length_stats']
                
                # Merge using Welford's algorithm for combining statistics
                duration_stats = merge_stats(duration_stats, file_duration_stats)
                pitch_stats = merge_stats(pitch_stats, file_pitch_stats)
                mel_stats = merge_stats(mel_stats, file_mel_stats)
                length_stats = merge_stats(length_stats, file_length_stats)
                
            except Exception as e:
                seg_path = future_to_path[future]
                print(f"Error processing {seg_path}: {e}")
                continue

    # Global statistics output
    stats = {
        "stats": {
            "duration_ms": duration_stats.finalize(),
            "pitches": pitch_stats.finalize(),
            "mel": mel_stats.finalize(),
            "sequence_length": length_stats.finalize(),
        }
    }
    os.makedirs("./metadata", exist_ok=True)
    with open("./metadata/dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Create phoneme index mapping
    phoneme_indexes = {
        "UNCOND": 0,
        "MUTE": 1,
        "PAD": 2
    }
    
    # Add unique phonemes starting from index 3
    sorted_phonemes = sorted(unique_phonemes)
    current_index = 3
    for phoneme in sorted_phonemes:
        if phoneme not in phoneme_indexes:
            phoneme_indexes[phoneme] = current_index
            current_index += 1
    
    # Save phoneme indexes
    phoneme_data = {"phonemes": phoneme_indexes}
    with open("./metadata/phoneme_indexes.json", "w", encoding="utf-8") as f:
        json.dump(phoneme_data, f, indent=2, ensure_ascii=False)

    phoneme_durations.sort(key=lambda x: x[1], reverse=True)
    logging.info("=== Top10 Longest Phonemes ===")
    for phoneme, dur in phoneme_durations[:10]:
        logging.info(f"{phoneme}: {dur:.1f} ms")
    logging.info("dataset_stats.json generated under ./metadata")
    logging.info(f"phoneme_indexes.json generated with {len(phoneme_indexes)} phonemes")


if __name__ == "__main__":
    main()
