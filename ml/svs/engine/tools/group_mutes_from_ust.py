from glob import glob
from os.path import join, sep

from tqdm.auto import tqdm

from engine.config.project import ProjectConfig

REMOVED_MARK = "_grouped_mute.lab"
MUTE_SET = {"sil", "sli", "pau"}

if __name__ == "__main__":
    files = sorted(glob(join(ProjectConfig.db_root, "**/*.lab"), recursive=True))

    for file in tqdm(files, desc="Grouping mutes"):
        try:
            # _TESTディレクトリのケア
            rate = float(file.split(sep=sep)[1].split("_")[2])
        except IndexError:
            rate = 1.0
        if file.endswith(REMOVED_MARK):
            continue

        try:
            with open(file, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            entries = []
            for line in lines:
                start_s, end_s, phoneme = line.split()
                entries.append((int(float(start_s) / rate), int(float(end_s) / rate), phoneme))

            output_lines = []
            grouping = False
            for start, end, phoneme in entries:
                if phoneme in MUTE_SET:
                    if not grouping:
                        grouping = True
                        grp_start, grp_end = start, end
                    else:
                        grp_end = end
                else:
                    if grouping:
                        output_lines.append(f"{grp_start} {grp_end} MUTE")
                        grouping = False
                    output_lines.append(f"{start} {end} {phoneme}")

            if grouping:
                output_lines.append(f"{grp_start} {grp_end} MUTE")

            output_path = file.replace(".lab", REMOVED_MARK)
            with open(output_path, "w") as f:
                f.write("\n".join(output_lines))

        except Exception as e:
            print("error file is", file)
            raise
