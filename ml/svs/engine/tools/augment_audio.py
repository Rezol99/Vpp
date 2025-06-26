import concurrent.futures
import glob
import json
import os
import shutil
from dataclasses import asdict

import librosa
import pyrubberband as pyrb
import soundfile as sf
from pydub import AudioSegment
from tqdm.auto import tqdm

from engine.config.audio import AudioConfig
from engine.config.train import TrainConfig
from engine.types.note_segment import MuteSegment, NoteSegment, PhonemeSegment
from engine.utils.time import hts_time_to_ms

RATES = [0.8, 0.9, 1, 1.1, 1.2]
PITCH = [-1, 0, 1]
MAX_WORKER = 16


def modify_singing_voice(
    input_path, output_path, speed, pitch_shift, sr=AudioConfig.audio.sampling_rate
):
    y, _ = librosa.load(input_path, sr=sr)
    y_stretched = pyrb.time_stretch(y, sr, rate=speed)
    y_final = pyrb.pitch_shift(y_stretched, sr, n_steps=pitch_shift)
    sf.write(output_path, y_final, sr)


def copy_dataset(rate, pitch):
    target_dir = f"./datasets/augment_rate_{rate}_pitch_{pitch}"
    print("Copying", target_dir)

    os.makedirs(target_dir, exist_ok=True)

    if os.listdir(target_dir):  # 既にファイルがある場合スキップ
        print(f"Skipping {target_dir} (already exists and is not empty)")
        return

    # `cp -r` を実行（ファイルコピー）
    os.system(f"cp -r ./default/* {target_dir}")
    print("Copied", target_dir)


def copy_default_datasets():
    tasks = [(rate, pitch) for rate in RATES for pitch in PITCH]
    max_workers = min(MAX_WORKER, len(tasks))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_dataset, rate, pitch) for rate, pitch in tasks]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in task: {e}")


def process_rate_pitch(rate: float, pitch: int):
    print(f"=== RATE {rate} PITCH {pitch} ===")
    if rate == 1 and pitch == 0:
        return

    target_dir = f"./datasets/augment_rate_{rate}_pitch_{pitch}"
    songs = glob.glob(f"{target_dir}/DATABASE/*")

    for song in tqdm(songs, desc=f"Processing rate: {rate} pitch: {pitch}"):
        name = song.split("/")[-1]

        # 音声処理
        wav = f"{song}/{name}.wav"
        modify_singing_voice(wav, wav, rate, pitch)

        # UST ファイル処理
        ust = f"{song}/{name}.ust"
        s = ""
        with open(ust, "r", encoding="shift_jis") as f:
            for line in f.readlines():
                if line.startswith("Tempo="):
                    try:
                        tempo = int(line.split("=")[-1].strip())
                    except ValueError:
                        tempo = float(line.split("=")[-1].strip())
                    tempo = tempo * rate
                    s += f"Tempo={tempo}\n"
                    continue
                if line.startswith("NoteNum="):
                    notenum = int(line.split("=")[-1].strip())
                    notenum += pitch
                    s += f"NoteNum={notenum}\n"
                    continue
                s += line
        with open(ust, "w", encoding="shift_jis") as f:
            f.write(s)


def modify_datasets():
    tasks = [
        (rate, pitch)
        for rate in RATES
        for pitch in PITCH
        if not (rate == 1 and pitch == 0)
    ]

    if len(tasks) == 0:
        print("Skip modify datasets")
        return

    # 最大24並列で並列実行
    max_workers = min(MAX_WORKER, len(tasks))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_rate_pitch, rate, pitch) for rate, pitch in tasks
        ]

        # エラー処理
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in task: {e}")


def remove_test_songs():
    songs = []
    names = []
    ignore_song = [song.replace("_TEST_", "") for song in TrainConfig.test_songs]

    for rate in RATES:
        for pitch in PITCH:
            if rate == 1 and pitch == 0:
                continue
            target_dir = f"./datasets/augment_rate_{rate}_pitch_{pitch}"
            songs_part = glob.glob(f"{target_dir}/DATABASE/*")
            names_part = [song.split("/")[-1] for song in songs_part]
            songs += songs_part
            names += names_part

    for i, name in enumerate(names):
        if name in ignore_song:
            path = songs[i]
            if os.path.exists(path):
                shutil.rmtree(path)


if __name__ == "__main__":
    copy_default_datasets()
    modify_datasets()
    remove_test_songs()
